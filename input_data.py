from datasets import convert_data
from preprocessing import vgg_preprocessing
from preprocessing import inception_preprocessing
from datasets import decode_tfrecord
import tensorflow as tf

slim = tf.contrib.slim


def read_image_and_label(dataset_dir, is_training=False):
    if is_training:
        dataset = decode_tfrecord.get_split(split_name='train', dataset_dir=dataset_dir)
    else:
        dataset = decode_tfrecord.get_split(split_name='validation', dataset_dir=dataset_dir)

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])

    return image, label


def read_image_and_label_test(dataset_dir, is_training=False):
    if is_training:
        dataset = decode_tfrecord.get_split(split_name='train', dataset_dir=dataset_dir)
    else:
        dataset = decode_tfrecord.get_split(split_name='validation', dataset_dir=dataset_dir)

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=False)
    [image, label] = provider.get(['image', 'label'])

    return image, label


def get_batch_images_and_label(dataset_dir, batch_size, num_classes, is_training=False, output_height=224,
                               output_width=224, num_threads=10):

    image, label = read_image_and_label(dataset_dir, is_training)
    image = inception_preprocessing.preprocess_image(image, output_height, output_width, is_training=is_training)

    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size,
        min_after_dequeue=batch_size - 3)

    labels = slim.one_hot_encoding(labels, num_classes)

    return images, labels


def load_batch_inception(dataset_dir, batch_size, num_classes, is_training=False, output_height=299, output_width=299,
                         num_threads=10):
    image, label = read_image_and_label(dataset_dir, is_training)
    image = inception_preprocessing.preprocess_image(image, output_height, output_width, is_training=is_training)

    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size,
        min_after_dequeue=batch_size)

    labels = slim.one_hot_encoding(labels, num_classes)

    return images, labels


def load_batch_inception_test(dataset_dir, batch_size, num_classes, is_training=False, output_height=299,
                              output_width=299, num_threads=1):
    image, label = read_image_and_label_test(dataset_dir, is_training)
    image = inception_preprocessing.preprocess_image(image, output_height, output_width, is_training=is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        capacity=5 * batch_size,
        num_threads=num_threads)

    labels = slim.one_hot_encoding(labels, num_classes)

    return images, labels
