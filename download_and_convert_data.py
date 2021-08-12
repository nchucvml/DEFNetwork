from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets import convert_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    'myown',
    'The name of the dataset to convert, one of "original_images".')

tf.app.flags.DEFINE_string(
    'dataset_source_train_dir',
    './dollars/train',
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_string(
    'dataset_source_test_dir',
    './dollars/test',
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_string(
    'dataset_destination_dir',
    './tfrecord',
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):

    if FLAGS.dataset_name == 'myown':
        convert_data.run(FLAGS.dataset_source_train_dir, FLAGS.dataset_source_test_dir, FLAGS.dataset_destination_dir)
    else:
        raise ValueError(
            'dataset_name [%s] was not recognized.' % FLAGS.dataset_name,
        )


if __name__ == '__main__':
    tf.app.run()
