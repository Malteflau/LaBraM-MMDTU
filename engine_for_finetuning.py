# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange
import numpy as np

def train_class_batch(model, samples, target, metadata, criterion, ch_names):
    outputs = model(samples, ch_names, metadata=metadata)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Handle both formats - with and without metadata
        if len(batch_data) == 3:
            samples, targets, metadata_batch = batch_data
        else:
            samples, targets = batch_data
            metadata_batch = None
            
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True)
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)
            
        # Process metadata
        if metadata_batch is not None:
            metadata = {
                k: v.to(device, non_blocking=True) 
                for k, v in metadata_batch.items()
            }
        else:
            metadata = None

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, metadata, criterion, input_chans)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, metadata, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
        
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def analyze_by_participant(data_loader, model, device, ch_names=None, is_binary=True):
    """
    Analyze model predictions grouped by individual participant to check for overfitting patterns.
    This function correctly extracts participant IDs from the DTU dataset.
    """
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    
    # switch to evaluation mode
    model.eval()
    
    # Dictionary to store per-participant statistics
    participant_stats = {}
    
    # For storing filenames from the dataset for debugging
    all_filenames = []
    
    # Get access to the dataset
    dataset = data_loader.dataset
    has_files = hasattr(dataset, 'files') and hasattr(dataset, 'valid_indices')
    
    # If we can access the dataset's files attribute, get all filenames
    if has_files:
        for idx in dataset.valid_indices:
            if idx < len(dataset.files):
                all_filenames.append(dataset.files[idx])
    
    print(f"Found {len(all_filenames)} files in dataset")
    print(f"Sample filenames: {all_filenames[:5] if all_filenames else 'None'}")
    
    # Track batch indices to map back to dataset indices
    batch_idx_global = 0
    
    for batch_data in data_loader:
        # Handle both formats - with and without metadata
        if len(batch_data) == 3:
            EEG, target, metadata_batch = batch_data
        else:
            EEG, target = batch_data
            metadata_batch = None
            
        EEG = EEG.float().to(device, non_blocking=True)
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
            
        # Process metadata
        if metadata_batch is not None:
            metadata = {
                k: v.to(device, non_blocking=True) 
                for k, v in metadata_batch.items()
            }
        else:
            metadata = None
        
        batch_size = EEG.shape[0]
        
        # Extract participant IDs
        # For DTU dataset, we need to map from batch indices to file indices
        participant_ids = []
        
        # If we have access to the dataset's files
        if has_files and all_filenames:
            for i in range(batch_size):
                # Calculate the global index in the dataset
                global_idx = batch_idx_global + i
                
                if global_idx < len(dataset.valid_indices):
                    file_idx = dataset.valid_indices[global_idx]
                    if file_idx < len(dataset.files):
                        # Get the filename from the dataset files
                        filename = dataset.files[file_idx]
                        
                        # Extract participant ID from filename (e.g., "301A_10_T13P.pkl")
                        parts = filename.split('_')
                        if len(parts) >= 1:
                            participant_id = parts[0]
                            participant_ids.append(participant_id)
                            continue
                
                # Fallback if we couldn't extract from filename
                participant_ids.append(f"unknown_{global_idx}")
        else:
            # If we can't get IDs from files, try to get from metadata
            if metadata_batch is not None and 'participant_id' in metadata_batch:
                for i in range(batch_size):
                    participant_ids.append(metadata_batch['participant_id'][i])
            else:
                # Ultimate fallback - just use batch indices
                participant_ids = [f"unknown_{batch_idx_global + i}" for i in range(batch_size)]
        
        # Compute model output
        with torch.cuda.amp.autocast():
            output = model(EEG, input_chans=input_chans, metadata=metadata)
        
        if is_binary:
            output_sigmoid = torch.sigmoid(output)
            predicted = (output_sigmoid > 0.5).float()
            probabilities = output_sigmoid.cpu().numpy()
        else:
            output_softmax = torch.softmax(output, dim=1)
            predicted = output.argmax(dim=1)
            probabilities = output_softmax.cpu().numpy()
        
        # Convert to numpy for analysis
        predictions = predicted.cpu().numpy()
        labels = target.cpu().numpy()
        
        # Update statistics for each participant in the batch
        for i, participant_id in enumerate(participant_ids):
            # Initialize participant stats if not present
            if participant_id not in participant_stats:
                participant_stats[participant_id] = {
                    'correct': 0,
                    'total': 0,
                    'tp': 0,
                    'tn': 0,
                    'fp': 0,
                    'fn': 0,
                    'label_dist': [0, 0],  # [count of 0s, count of 1s]
                    'pred_dist': [0, 0],   # [count of 0s, count of 1s]
                }
            
            # Get stats for this specific example
            if i < len(predictions):
                pred = predictions[i][0] if len(predictions[i].shape) > 0 else predictions[i]
                label = labels[i][0] if len(labels[i].shape) > 0 else labels[i]
                
                # Update statistics
                participant_stats[participant_id]['total'] += 1
                participant_stats[participant_id]['correct'] += int(pred == label)
                
                # Update label distribution
                label_idx = int(label) if is_binary else label
                if label_idx < len(participant_stats[participant_id]['label_dist']):
                    participant_stats[participant_id]['label_dist'][label_idx] += 1
                
                # Update prediction distribution
                pred_idx = int(pred) if is_binary else pred
                if pred_idx < len(participant_stats[participant_id]['pred_dist']):
                    participant_stats[participant_id]['pred_dist'][pred_idx] += 1
                
                # Update confusion matrix stats for binary classification
                if is_binary:
                    if pred == 1 and label == 1:
                        participant_stats[participant_id]['tp'] += 1
                    elif pred == 0 and label == 0:
                        participant_stats[participant_id]['tn'] += 1
                    elif pred == 1 and label == 0:
                        participant_stats[participant_id]['fp'] += 1
                    elif pred == 0 and label == 1:
                        participant_stats[participant_id]['fn'] += 1
        
        # Update global batch index
        batch_idx_global += batch_size
    
    # Calculate accuracy and other metrics for each participant
    for participant_id, stats in participant_stats.items():
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
        else:
            stats['accuracy'] = 0
            
        # Calculate precision, recall, F1
        if stats['tp'] + stats['fp'] > 0:
            stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp'])
        else:
            stats['precision'] = 0
            
        if stats['tp'] + stats['fn'] > 0:
            stats['recall'] = stats['tp'] / (stats['tp'] + stats['fn'])
        else:
            stats['recall'] = 0
            
        if stats['precision'] + stats['recall'] > 0:
            stats['f1'] = 2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall'])
        else:
            stats['f1'] = 0
    
    # Print summary of participant-level statistics
    print("\n----- Participant-level Analysis -----")
    print(f"Total participants analyzed: {len(participant_stats)}")
    
    # Calculate overall statistics
    total_correct = sum(stats['correct'] for stats in participant_stats.values())
    total_samples = sum(stats['total'] for stats in participant_stats.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    # Calculate variance in participant performance
    accuracies = [stats['accuracy'] for stats in participant_stats.values() if stats['total'] > 0]
    acc_variance = np.var(accuracies) if accuracies else 0
    print(f"Variance in participant accuracies: {acc_variance:.4f}")
    
    # Find outliers (participants with much higher/lower than average accuracy)
    if accuracies:
        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)
        outliers = [(pid, stats['accuracy']) for pid, stats in participant_stats.items() 
                   if abs(stats['accuracy'] - acc_mean) > 1.5 * acc_std and stats['total'] > 5]
        
        if outliers:
            print("\nOutlier participants (accuracy differs significantly from mean):")
            for pid, acc in sorted(outliers, key=lambda x: abs(x[1] - acc_mean), reverse=True):
                diff = acc - acc_mean
                print(f"  Participant {pid}: Accuracy {acc:.4f} ({diff:.4f} from mean)")
    
    # Check for bias in predictions
    total_label_dist = [sum(stats['label_dist'][i] for stats in participant_stats.values()) for i in range(2)]
    total_pred_dist = [sum(stats['pred_dist'][i] for stats in participant_stats.values()) for i in range(2)]
    
    print(f"\nTrue label distribution: {total_label_dist[0]} class 0, {total_label_dist[1]} class 1")
    print(f"Predicted distribution: {total_pred_dist[0]} class 0, {total_pred_dist[1]} class 1")
    
    if total_pred_dist[1] > 0.8 * total_samples or total_pred_dist[1] < 0.2 * total_samples:
        print("WARNING: Model predictions are heavily skewed toward one class!")
        print("This suggests the model may be overfitting or not learning meaningful patterns.")
    
    # Print per-participant statistics
    print("\nPer-participant statistics:")
    for participant_id, stats in sorted(participant_stats.items()):
        print(f"  Participant {participant_id}: Total={stats['total']} Acc={stats['accuracy']:.4f} "
              f"TP={stats['tp']} TN={stats['tn']} FP={stats['fp']} FN={stats['fn']} "
              f"Precision={stats['precision']:.4f} Recall={stats['recall']:.4f} F1={stats['f1']:.4f}")
    
    return participant_stats

@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    
    # Initialize counters for detailed metrics
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for step, batch_data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Handle both formats - with and without metadata
        if len(batch_data) == 3:
            EEG, target, metadata_batch = batch_data
        else:
            EEG, target = batch_data
            metadata_batch = None
            
        EEG = EEG.float().to(device, non_blocking=True)
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
            
        # Process metadata
        if metadata_batch is not None:
            metadata = {
                k: v.to(device, non_blocking=True) 
                for k, v in metadata_batch.items()
            }
        else:
            metadata = None
        
        # compute output
        with torch.cuda.amp.autocast():
            output = model(EEG, input_chans=input_chans, metadata=metadata)
            loss = criterion(output, target)
        
        if is_binary:
            # Apply sigmoid and threshold for binary prediction
            output_sigmoid = torch.sigmoid(output)
            predicted = (output_sigmoid > 0.5).float()
            
            # Calculate TP, TN, FP, FN for this batch
            batch_tp = ((predicted == 1) & (target == 1)).sum().item()
            batch_tn = ((predicted == 0) & (target == 0)).sum().item()
            batch_fp = ((predicted == 1) & (target == 0)).sum().item()
            batch_fn = ((predicted == 0) & (target == 1)).sum().item()
            
            # Update counters
            tp += batch_tp
            tn += batch_tn
            fp += batch_fp
            fn += batch_fn
            
            output = output_sigmoid.cpu()
        else:
            output = output.cpu()
        
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    
    # Calculate and print detailed confusion matrix metrics
    print(f'* Confusion Matrix Stats - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    
    # Calculate derived metrics
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f'* Precision: {precision:.4f}')
    else:
        precision = 0
        print('* Precision: N/A (no positive predictions)')
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f'* Recall: {recall:.4f}')
    else:
        recall = 0
        print('* Recall: N/A (no positive ground truth)')
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'* F1 Score: {f1:.4f}')
    else:
        f1 = 0
        print('* F1 Score: N/A (precision and recall are zero)')
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    
    # Add the detailed metrics to the return dict
    ret['tp'] = tp
    ret['tn'] = tn
    ret['fp'] = fp
    ret['fn'] = fn
    ret['precision'] = precision
    ret['recall'] = recall
    ret['f1'] = f1
    
    return ret