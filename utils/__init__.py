from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sys
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

from scipy.misc import imsave

from PIL import Image
import os

import sys
import tarfile

from six.moves import urllib

LABELS_FILENAME = 'labels.txt'

RANDOM_SEED = 4242

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
    }))


def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """Downloads the `tarball_url` and uncompresses it locally.

    Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
    """
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

    Returns:
    `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

    Returns:
    A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read()
    lines = lines.split(b'\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(b':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names


class Dataset(object):
    def __init__(self, opts):
        print('dataset initializing ... ...')
        self.opts = opts

        file_list = os.listdir(os.path.join(self.opts.ROOT_DIR, self.opts.DATASET_DIR))
        imgs_file = [os.path.join(os.path.join(self.opts.ROOT_DIR, self.opts.DATASET_DIR), filename) for filename in
                     file_list]
        imgs = dict()
        imgs['data'] = []

        for index, file in enumerate(imgs_file):
            img = Image.open(file)
            img = np.array(img)
            imgs['data'].append(img)
            print(index,self.opts.TEST_SIZE)
            if index>self.opts.TEST_SIZE:
                break
            
        imgs['data'] = np.array(imgs['data'])
        
        print('trian images shape = ', self.opts.TRAIN_SIZE)
        self.images = imgs['data'].astype('uint8')
        self.test_images = np.zeros((self.opts.TEST_SIZE, self.opts.IMG_H, self.opts.IMG_W, 3))
        self.images = self.images.astype('float')

        np.random.shuffle(self.images)
        self.test_images = self.images[:self.opts.TEST_SIZE, :, :, :]

        print('test images shape = ', self.test_images.shape)
        self.save_batch_images(self.test_images, [self.opts.GRID_H, self.opts.GRID_W], "target.jpg")


    def convert_to_tfrecords(self, name='train', shuffling=True):
        """Runs the conversion operation.
        """
        print("begin to convert ... ... ")

        if not tf.gfile.Exists(self.opts.TFRECORD_DIR):
            tf.gfile.MakeDirs(self.opts.TFRECORD_DIR)

        # Dataset filenames, and shuffling.
        path = os.path.join(self.opts.ROOT_DIR,self.opts.DATASET_DIR)
        print(path)
        filenames = sorted(os.listdir(path))
        #filenames = [os.path.join(path,file) for file in filenames]

        if shuffling:
            random.seed(RANDOM_SEED)
            random.shuffle(filenames)

        # Process dataset files.
        i = 0
        fidx = 0
        with tf.Session() as sess:
            while i < len(filenames):
                # Open new TFRecord file.
                tf_filename = self._get_output_filename(name, fidx)
                with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                    j = 0
                    while i < len(filenames) and j < self.opts.SAMPLES_PER_TFRECORD_FILES:
                        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
                        sys.stdout.flush()
    
                        filename = filenames[i]
                        img_name = filename[:-4]
                        self._add_to_tfrecord(img_name, tfrecord_writer)
                        i += 1
                        j += 1
                    fidx += 1
    
            print('\nFinished converting the dataset to tfrecord !')

    def get_batch(self, shuffle=True):

        tfrecordfile_list = os.listdir(os.path.join(self.opts.ROOT_DIR, self.opts.TFRECORD_DIR))
        tfrecord_file_name = [os.path.join(os.path.join(self.opts.ROOT_DIR, self.opts.TFRECORD_DIR), filename) for filename in
                     tfrecordfile_list]

        reader = tf.TFRecordReader

        keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, ''),
            "image/format": tf.FixedLenFeature((), tf.string, 'jpeg')}

        items_to_handlers = {
            "image": slim.tfexample_decoder.Image(shape=[self.opts.IMG_H, self.opts.IMG_W, 3]),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers
        )
        item_description = {"image": "a color image"}

        dataset = slim.dataset.Dataset(
            data_sources=tfrecord_file_name,
            reader=reader,
            decoder=decoder,
            items_to_descriptions={},
            num_samples=self.opts.SAMPLES_PER_TFRECORD_FILES,
            items_to_description=item_description
        )

        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                  num_readers=4,
                                                                  common_queue_capacity=10 * self.opts.BATCH_SIZE,
                                                                  common_queue_min=5 * self.opts.BATCH_SIZE,
                                                                  shuffle=shuffle)

        row_image = provider.get(['image'])
        row_image = tf.to_float(row_image)
        image = tf.div(tf.reshape(row_image, [self.opts.IMG_H,self.opts.IMG_W, 3]),255)
        #image = tf.image.resize_images(image, [self.opts.IMG_H, self.opts.IMG_W], method=0)
        print("image shape", image.get_shape())
        '''a_batch = tf.train.shuffle_batch([image],
                                         batch_size=self.opts.BATCH_SIZE,
                                         capacity=3 * self.opts.BATCH_SIZE,
                                         min_after_dequeue=1 * self.opts.BATCH_SIZE)'''
        a_batch = tf.train.batch([image],
                                         batch_size=self.opts.BATCH_SIZE,
                                         capacity=3 * self.opts.BATCH_SIZE,
                                         num_threads=4)
        print("batch shape", a_batch.get_shape())

        return a_batch

    def save_batch_images(self, images, grid, img_file, normalized=False):
        h = images.shape[1]
        w = images.shape[2]
        # if normalized and self.opts.model != 'gan':
            	# images = images * 255.0
        # elif normalized and self.opts.model == 'gan':
            	# images = images * 127.5 + 127.5
        num = images.shape[0]

        imgs = np.zeros((h * grid[0], w * grid[1], self.opts.CHANNELS))

        for idx, image in enumerate(images):
            i = idx % grid[1]
            j = idx // grid[1]
            imgs[i * w:w * (i + 1), j * h:(j + 1) * h, :] = image

        img_file_path = os.path.join(self.opts.ROOT_DIR, self.opts.SAMPLE_DIR, img_file)
        imsave(img_file_path, imgs)

    # with open(self.opts.root_dir+self.opts.sample_dir+img_file.split('.')[0]+'-pkl', 'wb') as f:
    # 	cPickle.dump(imgs, f)
    def _process_image(self, image_name):
        """Process a image

        Args:
          filename: string, path to an image file e.g., '/path/to/example.JPG'.
          coder: instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
          image_buffer: string, JPEG encoding of RGB image.
          height: integer, image height in pixels.
          width: integer, image width in pixels.
        """

    # Read the image file.
        filename = os.path.join(
            os.path.join(self.opts.ROOT_DIR,self.opts.DATASET_DIR),
            image_name + '.jpg')

        print("\nprocessing %s"%filename)

        image_data = tf.gfile.FastGFile(filename, 'rb').read()
        '''image_data = tf.image.decode_jpeg(image_data)
        
        image_data = tf.image.resize_images(image_data, [self.opts.IMG_H, self.opts.IMG_W], method=0)
        image_data = image_data / 255.0
        
        
        image_data = tf.image.convert_image_dtype(image_data, dtype=tf.uint8)
        image_data = tf.image.encode_jpeg(image_data).eval()'''

        #print("\nprocess %s done "%filename)
        return image_data

    def _convert_to_example(self, image_data):
        """Build an Example proto for an image example.

        Args:
          image_data: string, JPEG encoding of RGB image;

        Returns:
          Example proto
        """

        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
        return example

    def _add_to_tfrecord(self, image_name, tfrecord_writer):
        """Loads data from image files and add to a TFRecord.

        Args:
          name: Image name to add to the TFRecord;
          tfrecord_writer: The TFRecord writer to use for writing.
        """
        image_data = self._process_image(image_name)
        image_data = bytes(image_data)
        example = self._convert_to_example(image_data)
        tfrecord_writer.write(example.SerializeToString())

    def _get_output_filename(self, name, idx):
        return '%s/%s_%03d.tfrecord' % (os.path.join(self.opts.ROOT_DIR,self.opts.TFRECORD_DIR), name, idx)
