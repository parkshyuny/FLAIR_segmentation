import numpy as np
import tensorflow as tf

class Logger():
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(
        self, 
        loss: list, 
        step: int, 
        prefix=""
    ):
        tf.summary.scalar(prefix + "loss", np.mean(loss), step)