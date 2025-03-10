import tensorflow as tf

# Path to the events file
events_file = 'log/finetune_dtu_base/friendship/events.out.tfevents.1741177347.n-62-20-1'

# Create an event accumulator
for summary in tf.train.summary_iterator(events_file):
    for v in summary.summary.value:
        print(f"Tag: {v.tag}, Value: {v.simple_value}")
