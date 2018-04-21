import numpy as np
import tensorflow as tf
import os
import tensorflow.contrib as contrib
import cv2 as cv
import net_visualize
import time

# global parameters
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
batch_size = 50
NUM_CHANNEL = 1
noise_factor = 0.01  ## (0~1)
NUM_FILTER = 32
lambd = 0.0001 ## (0~1)
FEATURE_FOLDER = './feature_map/'
tex_layers = ['pool3_2', 'pool2_2', 'pool1_2', 'conv1_1']

class My_net():
    def __init__(self):
        pass

    def buildnet(self,input_tensor):
        variable_batch_size = tf.shape(input_tensor)[0]

        # layer1
        # 原始输入是32×32*1
        self.conv1_1 = tf.layers.conv2d(inputs=input_tensor, filters=64, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu, name='conv1_1')

        ## 池化输出16*16*64
        self.pool1_2 = tf.layers.max_pooling2d(inputs=self.conv1_1, pool_size=[2, 2], strides=[2,2], name = 'pool1')

        # layer2
        # 输入16*16*64
        self.conv2_1 = tf.layers.conv2d(inputs=self.pool1_2, filters=64, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu, name = 'conv2_1')
        # 输出8×8×64
        self.pool2_2 = tf.layers.max_pooling2d(inputs=self.conv2_1, pool_size=[2, 2], strides=[2,2], name = 'pool2')

        # layer 3
        # 输入8×8×64
        self.conv3_1 = tf.layers.conv2d(inputs=self.pool2_2, filters=32, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu, name = 'conv3_1')
        # 输出4×4×32
        self.pool3_2 = tf.layers.max_pooling2d(inputs=self.conv3_1, pool_size=[2, 2], strides=[2,2], name = 'pool3')

        # deconv layer
        # 输出8×8×32
        self.deconv3_1 = tf.layers.conv2d_transpose(inputs=self.pool3_2, filters=32, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')
        # 输出16×16×64
        self.deconv2_1 = tf.layers.conv2d_transpose(inputs=self.deconv3_1, filters=64, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')
        # 输出32×32×64
        self.deconv1_1 = tf.layers.conv2d_transpose(inputs=self.deconv2_1, filters=64, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')

        # conv layer :
        # 输出32×32×1
        self.final = tf.layers.conv2d(inputs=self.deconv1_1, filters=1, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu)
        return self.final

    def loss_function(self,predicts,labels):
        loss = tf.reduce_mean(tf.pow(tf.subtract(predicts, labels), 2.0))  #此处tf.pow为幂运算函数
        return loss

        # origin = labels
        # new = predicts
        # origin = tf.layers.conv2d(inputs=origin, filters=16, kernel_size=[1, 1], padding='same',
        #                            activation=tf.nn.relu)
        # origin = tf.layers.conv2d(inputs=origin, filters=16, kernel_size=[1, 1], padding='same',
        #                            activation=tf.nn.relu)
        # origin = tf.layers.conv2d(inputs=origin, filters=16, kernel_size=[1, 1], padding='same',
        #                            activation=tf.nn.relu)
        # new = tf.layers.conv2d(inputs=new, filters=16, kernel_size=[1, 1], padding='same',
        #                            activation=tf.nn.relu)
        # new = tf.layers.conv2d(inputs=new, filters=16, kernel_size=[1, 1], padding='same',
        #                            activation=tf.nn.relu)
        # new = tf.layers.conv2d(inputs=new, filters=16, kernel_size=[1, 1], padding='same',
        #                            activation=tf.nn.relu)
        # shape = origin.get_shape().as_list()
        # N = shape[3]
        # M = shape[1]*shape[2]
        # F = tf.reshape(origin,(-1,N))
        # Gram_o = (tf.matmul(tf.transpose(F),F)/(N*M))
        # F_t = tf.reshape(new,(-1,N))
        # Gram_n = tf.matmul(tf.transpose(F_t),F_t)/(N*M)
        # loss = tf.nn.l2_loss((Gram_o-Gram_n))/2
        # return loss

    def loss_function_corr(self, predicts, labels):
        '''考虑两幅图像的相关系数
        a = a - mean2(a);
        b = b - mean2(b);
        r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));
        '''
        shape =  predicts.get_shape().as_list()
        n = tf.shape(predicts)[0] # how many patches?

        axis = list(range(1, len(shape)))  # 1,2,3, no 0
        m1 = tf.reduce_mean(predicts, axis)
        m2 = tf.reduce_mean(labels, axis)

        m1 = tf.reshape(m1, shape=[n,1,1,1])
        m2 = tf.reshape(m2, shape=[n, 1, 1, 1])
        shape_tile = [1, *shape[1:]]
        a = predicts - tf.tile(m1,shape_tile)
        b = labels - tf.tile(m2,shape_tile)
        corr = tf.reduce_sum(tf.multiply(a, b), axis) \
               / tf.sqrt(tf.reduce_sum(tf.pow(a, 2), axis)
                       * tf.reduce_sum(tf.pow(b, 2), axis))

        loss_corr = 1.0 / tf.reduce_mean(corr) # inverse of correlation
        loss_mse = tf.reduce_mean(tf.pow(tf.subtract(predicts, labels), 2.0)) # mean squared error

        tf.add_to_collection('loss_corr', loss_corr)
        # tf.add_to_collection('losses', loss_mse)
        loss = tf.add_n(tf.get_collection('loss_corr'))#losses

        return loss


    def patch_list2matrix(self,batch_list):# batches is list
        n = len(batch_list)
        batches = np.zeros(dtype=np.float32, shape=(n,INPUT_HEIGHT, INPUT_WIDTH))
        for i in range(n):
            batches[i,:,:] = batch_list[i]
        training_data = np.reshape(batches, newshape=[n, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL])
        return training_data

    def tensorlist2patches(self, batches):
        patches = []
        n = len(batches)
        for i in range(n):
            num_im = batches[i].shape[0] # how many images are there in current batch
            for j in range(num_im):
                patch = np.reshape(batches[i][j], newshape=(INPUT_HEIGHT, INPUT_WIDTH))
                patches.append(patch )
        return patches

    def train(self,sess, training_data, optimizer, loss, x, y, epochs = 100):
        # reshape data
        n = len(training_data)
        training_data = self.patch_list2matrix(training_data)

        for j in range(epochs+1):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                noise_x = mini_batch + noise_factor * np.random.randn(*mini_batch.shape) * 255
                noise_x = np.clip(noise_x, 0., 255.)
                _,loss_val = sess.run([optimizer, loss], feed_dict={x: noise_x, y: mini_batch})
            print("Epoch {0} complete".format(j))

            if j % 10 == 0 or j == epochs:
                print('After training steps, loss on training batch is: %g ' % (loss_val))

    def evaluate(self, sess, patches, predicts, loss2, x):
        n = len(patches)
        testing_data = self.patch_list2matrix(patches)
        reconstructed_batches = []
        mini_batches = [testing_data[k:k + batch_size] for k in range(0, n, batch_size)]
        for mini_batch in mini_batches:
            recontructed_batch, loss_val = sess.run([predicts,loss2], feed_dict={x: mini_batch}) # recontructed_batch 是4维矩阵 ： batch_size * H * W * 1
            reconstructed_batches.append(recontructed_batch)
            reconst_patches = self.tensorlist2patches(reconstructed_batches)
            print(n)
        return reconst_patches

    def feature_train(self, sess, patches, x):
        num = len(patches)
        conv_list = []
        feature_sum = []
        conv_list.append(self.conv1_1)
        conv_list.append(self.pool1_2)
        conv_list.append(self.pool2_2)
        conv_list.append(self.pool3_2)
        dim = len(conv_list)

        test_patch = patches[0].reshape((1,32,32,1))
        feature_sum = sess.run(conv_list, feed_dict={x: test_patch})

        for n in range(num-1):
            test_patch = patches[n+1].reshape((1,32,32,1))
            feature_one = sess.run(conv_list, feed_dict={x: test_patch})
            for d in range(dim):
                #start_time = time.time()
                #net_visualize.plot_conv_output(visual_patch[d], FEATURE_FOLDER, 'featrue_map_'+str(start_time+d))
                #print(visual_patch[d].shape)
                feature_sum[d] = (feature_one[d] + feature_sum[d])
            print('number %d patch have finished' % (n))
        feature_mean = [i/num for i in feature_sum]
        return feature_mean

    def feature_test(self, sess, patch, x):
        conv_list = []
        feature_sum = []
        conv_list.append(self.conv1_1)
        conv_list.append(self.pool1_2)
        conv_list.append(self.pool2_2)
        conv_list.append(self.pool3_2)
        dim = len(conv_list)

        test_patch = patch.reshape((1,32,32,1))
        feature_map = sess.run(conv_list, feed_dict={x: test_patch})
        return feature_map