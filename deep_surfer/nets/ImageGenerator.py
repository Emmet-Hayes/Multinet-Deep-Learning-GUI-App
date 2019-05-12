import os #gives us access to system i/o and files
import numpy as np #better numbers for python
import random
import scipy.misc
import cv2 #image manipulation imports
from PIL import Image
import tensorflow as tf #the deep learning good stuff
#from PyQt5.QtWidgets import QFileDialog #gui system file access

class ImageGenerator:

  @staticmethod #resizes the folders of images to a square with side lengths passed
  def resize(notepadWidget, side_length=256):
    #src = QFileDialog.getExistingDirectory(notepadWidget, 
    #  "Hint: Open a directory with image files in it", os.getenv('HOME'))
    #dst = QFileDialog.getExistingDirectory(notepadWidget, 
    #  "Hint: Open a directory to save resized images in", os.getenv('HOME'))
    if not os.path.isdir(dst):
      os.mkdir(dst)
    for each in os.listdir(src):
      img = cv2.imread(os.path.join(src,each))
      img = cv2.resize(img,(side_length,side_length))
      cv2.imwrite(os.path.join(dst,each), img)

  @staticmethod #removes the alpha channel from a folder of images
  def rgba2rgb(notepadWidget):
    #src = QFileDialog.getExistingDirectory(notepadWidget, 
    #  "Hint: Open a directory with rgba image files in it", os.getenv('HOME'))
    #dst = QFileDialog.getExistingDirectory(notepadWidget, 
    #  "Hint: Open a directory to save rgb images in", os.getenv('HOME'))
    if not os.path.isdir(dst):
      os.mkdir(dst)
    for each in os.listdir(src):
      png = Image.open(os.path.join(src,each))
      if png.mode == 'RGB':
        png.load() # required for png.split()
        background = Image.new("RGBA", png.size, (0,0,0,0))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
      else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')

  @staticmethod #
  def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 

  @staticmethod #
  def process_data(notepadWidget, height=128, width=128, batch=64, channel=3):   
    #image_dir = QFileDialog.getExistingDirectory(notepadWidget, 
    #  "Hint: Open a directory with resized rgb image files in it")
    images = []
    try:
      for each in os.listdir(image_dir):
        images.append(os.path.join(image_dir,each))
    except FileNotFoundError:
      return 
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer([all_images])
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = channel)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    size = [height, width]
    image = tf.image.resize_images(image, size)
    image.set_shape([height,width,channel])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    images_batch = tf.train.shuffle_batch([image], batch_size = batch,
      num_threads = 4, capacity = 200 + 3* batch, min_after_dequeue = 200)
    num_images = len(images)
    return images_batch, num_images

  @staticmethod #
  def generator(input, random_dim, is_train, channel_count=3, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = channel_count  # RGB image
    with tf.variable_scope('gen') as scope:
      if reuse:
        scope.reuse_variables()
      w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
      b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
      flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
      conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1') #Convolution, bias, activation, repeat! 
      bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
      act1 = tf.nn.relu(bn1, name='act1')
      conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME", # 8*8*256
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv2')
      bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
      act2 = tf.nn.relu(bn2, name='act2')
      conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME", # 16*16*128
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv3')
      bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
      act3 = tf.nn.relu(bn3, name='act3')
      conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME", # 32*32*64
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv4')
      bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
      act4 = tf.nn.relu(bn4, name='act4')
      conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME", # 64*64*32
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv5')
      bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
      act5 = tf.nn.relu(bn5, name='act5')
      conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME", #128*128*3
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv6')
      # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
      act6 = tf.nn.tanh(conv6, name='act6')
      return act6

  @staticmethod #
  def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
      if reuse:
        scope.reuse_variables()
      conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME", #128*128*3
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               name='conv1')
      bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
      act1 = ImageGenerator.lrelu(conv1, n='act1')
      conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME", #64*64*32
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               name='conv2')
      bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
      act2 = ImageGenerator.lrelu(bn2, n='act2')
      conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME", #32,32,64
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               name='conv3')
      bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
      act3 = ImageGenerator.lrelu(bn3, n='act3')
      conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME", #16*16*128
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               name='conv4')
      bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4') #8*8*256
      act4 = ImageGenerator.lrelu(bn4, n='act4')
      dim = int(np.prod(act4.get_shape()[1:])) # start from act4
      fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
      w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
      b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0))
      # wgan just get rid of the sigmoid
      logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
      # dcgan
      #acted_out = tf.nn.relu(logits)
      acted_out = tf.nn.sigmoid(logits)
      return acted_out #, logits #, acted_out

  @staticmethod
  def train(notepadWidget, height=128, width=128, channel=3, batch_size=64, epoch=1000, random_dim=100, 
    learn_rate=2e-4, clip_weights=0.01, d_iters=5, g_iters=1, save_ckpt_rate=500, save_img_rate=50):
    version = "generated"
    gen_image_path = './' + version
    with tf.variable_scope('input'):
      real_image = tf.placeholder(tf.float32, shape = [None, height, width, channel], name='real_image')
      random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
      is_train = tf.placeholder(tf.bool, name='is_train')
    fake_image = ImageGenerator.generator(random_input, random_dim, is_train, channel)
    real_result = ImageGenerator.discriminator(real_image, is_train)
    fake_result = ImageGenerator.discriminator(fake_image, is_train, reuse=True)
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    d_clip = [v.assign(tf.clip_by_value(v, 0 - clip_weights, clip_weights)) for v in d_vars] # clip discriminator weights
    image_batch, samples_num = ImageGenerator.process_data(notepadWidget)
    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    save_path = saver.save(sess, "./ImageGeneratorModel/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./ImageGeneratorModel/' + version)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, epoch))
    print('start training...')
    for i in range(epoch):
      sess.run(tf.local_variables_initializer())
      print("Running epoch {}/{}...".format(i, epoch))
      for j in range(batch_num):
        print(j)
        train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
        for k in range(d_iters): # Update the discriminator
          print(k, ' aligning discriminatory chakras...')
          train_image = sess.run(image_batch)
          sess.run(d_clip) #wgan clip weights
          _, dLoss = sess.run([trainer_d, d_loss], 
                            feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
        for k in range(g_iters): # Update the generator
          _, gLoss = sess.run([trainer_g, g_loss],
                              feed_dict={random_input: train_noise, is_train: True})
        # print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)
      if i%save_ckpt_rate == 0: # save check point every x epoch
        if not os.path.exists('./ImageGeneratorModel/' + version):
          os.makedirs('./ImageGeneratorModel/' + version)
        saver.save(sess, './ImageGeneratorModel/' +version + '/' + str(i))  
      if i%save_img_rate == 0: # save images every y epoch
        if not os.path.exists(gen_image_path):
          os.makedirs(gen_image_path)
        sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
        imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
      print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)
    print("image generator is done! navigate back to the main window")