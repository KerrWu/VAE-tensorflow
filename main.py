import os
import tensorflow as tf

import utils

import nnet

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

# Model
tf.app.flags.DEFINE_string('model', 'vae', """vae or gan""")
tf.app.flags.DEFINE_float('GPU_PERCENTAGE','0.8','the percenage opf gpu utility')

# Tfrecords
tf.app.flags.DEFINE_integer('SAMPLES_PER_TFRECORD_FILES', 256,
                            """Number of images per tfrecord file""")

tf.app.flags.DEFINE_string('TFRECORD_DIR', 'tfrecords',
                           """The dir to save tfrecord files""")
tf.app.flags.DEFINE_boolean('CONVERT', False, """Convert image to tfrecord""")

# Data
tf.app.flags.DEFINE_string('ROOT_DIR', '/home/wz/HD1/LiYi/VAE_GAN', """Base Path""")
tf.app.flags.DEFINE_string('DATASET_DIR', 'data', """Path to data""")
tf.app.flags.DEFINE_integer('IMG_H', 300, """Shape of image height""")
tf.app.flags.DEFINE_integer('IMG_W', 300, """Shape of image width""")
tf.app.flags.DEFINE_integer('CHANNELS', 3, """Number of input channels of images""")
tf.app.flags.DEFINE_string('DATASET_NAME', "mnist", """mnist or CIFAR""")

# Training
tf.app.flags.DEFINE_integer('NUM_GPU', 3, """number of gpu""")
tf.app.flags.DEFINE_integer('MAX_TO_KEEP', 50, """max number of ckpts to keep""")
tf.app.flags.DEFINE_integer('BATCH_SIZE', 16, """Batch size""")
tf.app.flags.DEFINE_integer('MAX_ITERATIONS', 100, """Max iterations for training""")
tf.app.flags.DEFINE_integer('DECAY_AFTER_GS', 10000, """Decay learning after global step""")
tf.app.flags.DEFINE_integer('CKPT_FRQ', 5, """Frequency at which to checkpoint the model""")
tf.app.flags.DEFINE_integer('TRAIN_SIZE', 67066, """The total training size""")

tf.app.flags.DEFINE_integer('GENERATE_FRQ', 5, """The frequency with which to sample images""")
tf.app.flags.DEFINE_integer('LOG_DECAY', 1, """The frequency with which to log""")

tf.app.flags.DEFINE_integer('TEST_SIZE', 16, """Number of images to sample during test phase""")
tf.app.flags.DEFINE_integer('DISPLAY', 50, """Display log of progress""")
tf.app.flags.DEFINE_float('LR_DECAY', 0.9, """Learning rate decay factor""")
tf.app.flags.DEFINE_float('BASE_LR', 1e-4, """Base learning rate for VAE""")
tf.app.flags.DEFINE_float('MIN_LR', 1e-6, """Base learning rate for VAE""")
tf.app.flags.DEFINE_float('D_base_lr', 2e-4, """Base learning rate for Discriminator""")
tf.app.flags.DEFINE_float('G_base_lr', 2e-4, """Base learning rate for Generator""")
tf.app.flags.DEFINE_float('D_LAMDA', 1, """How much to weigh in Decoder loss""")
tf.app.flags.DEFINE_float('G_LAMDA', 1, """How much to weigh in Generator loss""")
tf.app.flags.DEFINE_boolean('TRAIN', True, """Training or testing/with different network configuration""")
tf.app.flags.DEFINE_boolean('TRAINTRAIN', False, """Training or testing the encoder and decoder""")
tf.app.flags.DEFINE_boolean('TRAIN_GMM', False, """Training or testing GMM""")


# Architecture
tf.app.flags.DEFINE_integer('ENCODE_VECTOR_SIZE', 512, """Encoder vector size for VAE""")
tf.app.flags.DEFINE_integer('code_len', 100, """Latent code length in case of GAN""")
tf.app.flags.DEFINE_integer('DIMS', 32,
                            """Number of kernels for the first convolutional lalyer in the network for GAN/VAE""")
tf.app.flags.DEFINE_integer('label_len', 1, """Number of output units in discriminator""")
tf.app.flags.DEFINE_boolean('use_labels', False, """Should use labels in cross-entropy for GANs ?""")
tf.app.flags.DEFINE_integer('NUM_RESIDUAL_UNITS', 3, """Number of layers per resiual units""")

# Model Saving
tf.app.flags.DEFINE_boolean('IS_TRAINING', True, """If there are model to be restored""")
tf.app.flags.DEFINE_string('CKPT_DIR', "ckpt", """Checkpoint Directory""")
tf.app.flags.DEFINE_string('SAMPLE_DIR', "imgs", """Generate sample images""")
tf.app.flags.DEFINE_string('SUMMARY_DIR', "summary", """Summaries directory""")
tf.app.flags.DEFINE_integer('GRID_H', 4, """Grid height while saving images""")
tf.app.flags.DEFINE_integer('GRID_W', 4, """Grid width while saving images""")
tf.app.flags.DEFINE_boolean('RESTORE', True, """If there are model to be restored""")
tf.app.flags.DEFINE_string('RESTORE_MODEL', "/home/wz/HD1/LiYi/VAE_GAN/ckpt/50",
                           """the model path to be restored""")


def main(_):
    
    print('begin ... ...')

    if FLAGS.model == "vae":
        print('build VAE model')
        net = nnet.VAE(FLAGS)
    else:
        net = nnet.GAN(FLAGS)
    
    
    if FLAGS.CONVERT:
        print('converting data to tfrecord file ... ... ')
        dataset = utils.Dataset(FLAGS)
        dataset.convert_to_tfrecords()
        
    if FLAGS.TRAINTRAIN:
        print('Training the network ... ...')
        net.main()
        print('Done training the network...\n')
    else:
        
        if FLAGS.TRAIN_GMM:
            print("Train GMM")
            net.cluster("/home/wz/HD1/LiYi/VAE_GAN/data")
            print("Done")
        else:
            
            print("Clustering the data")
            net.cluster("/home/wz/HD1/LiYi/VAE_GAN/data")
            print("Done cluster the data")


    


if __name__ == '__main__':
    try:
        tf.app.run()
    except Exception as E:
        print(E)
