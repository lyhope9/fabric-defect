import numpy as np
import tensorflow as tf
import os
import cv2

INPUT_HEIGHT = 32
INPUT_WIDTH = 32
batch_size = 100
count = 0
groups=256

class Network():

    def __init__(self):
        self.learning_rate=0.0001
        self.Steps=10000

    def build_network(self,inputs):
        conv1_1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu)
        pool1_1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], strides=2)
        conv1_2 = tf.layers.conv2d(inputs=pool1_1, filters=64, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu)
        pool1_2 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)
        conv1_3 = tf.layers.conv2d(inputs=pool1_2, filters=32, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu)
        pool1_3 = tf.layers.max_pooling2d(inputs=conv1_3, pool_size=[2, 2], strides=2)

        deconv1_1 = tf.layers.conv2d_transpose(inputs=pool1_3, filters=32, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')
        deconv1_2 = tf.layers.conv2d_transpose(inputs=deconv1_1, filters=64, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')
        deconv1_3 = tf.layers.conv2d_transpose(inputs=deconv1_2, filters=64, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')
        final = tf.layers.conv2d(inputs=deconv1_3, filters=3, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu)

    #     weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=0.1, name='weight_1'))
    #     bias_1 = tf.Variable(tf.constant(0.0, shape=[32], name='bias_1'))
    #     weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1, name='weight_2'))
    #     bias_2 = tf.Variable(tf.constant(0.0, shape=[64], name='bias_2'))
    #     weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64,32], stddev=0.1, name='weight_3'))
    #     bias_3 = tf.Variable(tf.constant(0.0, shape=[128], name='bias_3'))
    #
    #     deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.1), name='deconv_weight_1')
    #     deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=0.1), name='deconv_weight_2')
    #     deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name='deconv_weight_3')
    #
    #     conv1 = tf.nn.conv2d(input=input, filter=weight_1, strides=[1, 1, 1, 1], padding='SAME')
    #     conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')
    #     acti1 = tf.nn.relu(conv1, name='acti_1')
    #     pool1 = tf.nn.max_pool(value=acti1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
    #
    #     conv2 = tf.nn.conv2d(input=pool1, filter=weight_2, strides=[1, 1, 1, 1], padding='SAME')
    #     conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')
    #     acti2 = tf.nn.relu(conv2, name='acti_2')
    #     pool2 = tf.nn.max_pool(value=acti2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')
    #
    #     conv3 = tf.nn.conv2d(input=pool2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME')
    #     conv3 = tf.nn.bias_add(conv3, bias_3, name='conv_3')
    #     acti3 = tf.nn.relu(conv3, name='acti_3')
    #     pool3 = tf.nn.max_pool(value=acti3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_3')
    #
    #     deconv1 = tf.nn.conv2d_transpose(value=pool3, filter=deconv_weight_1, output_shape=[batch_size, 4, 4, 32],strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')
    #     deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=[batch_size, 8, 8, 64],strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')
    #     deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=[batch_size, 16, 16, 64],strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')
    #
    #     weight_final = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 3], stddev=0.1, name='weight_final'))
    #     bias_final = tf.Variable(tf.constant(0.0, shape=[3], name='bias_final'))
    #     conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')
    #     conv_final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')
    #
        return final


    def loss_function(self,predicts,labels):
        loss = tf.reduce_mean(tf.pow(tf.subtract(predicts, labels), 2.0))  #此处tf.pow为幂运算函数
        return loss


    def get_data(self,train_data):
        img_batch = np.zeros((batch_size, INPUT_WIDTH, INPUT_HEIGHT, 3))
        num = 0
        for i in range(batch_size):
            global count
            img_batch[num, :, :, :] = train_data[count]
            num = num + 1
            count = count + 1
            if count >= len(train_data):
                np.random.shuffle(train_data)
                count = 0
        return img_batch


    def train(self,train_data):
        input = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_WIDTH, INPUT_HEIGHT, 3))
        predicts = self.build_network(input)
        loss = self.loss_function(predicts, input)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Training start......')
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state("train_data\model_file")
            if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)

            for i in range(self.Steps):
                img_batch = self.get_data(train_data)
                sess.run(train_step, feed_dict={input: img_batch})

                if i % 100 == 0 or i == self.Steps - 1:
                    total_loss = sess.run(loss, feed_dict={input: img_batch})
                    print('Step:%d  The total loss is: ' % i)
                    print(total_loss)

                if (i > 0 and i % 1000 == 0) or i == self.Steps - 1:
                    model_path = 'train_data\model_file\model_' + str(i) + '.ckpt'
                    saver.save(sess, model_path)
                    print('Saving the train model file: ' + os.path.basename(model_path))

    def Imgtest(self,img):
        image=np.zeros((1,INPUT_WIDTH, INPUT_HEIGHT, 3))
        image[0]=img
        input = tf.placeholder(dtype=tf.float32, shape=(1,INPUT_WIDTH, INPUT_HEIGHT, 3))
        predicts = self.build_network(input)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state("train_data\model_file")
            if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            output=sess.run(predicts, feed_dict={input: image})
            outimg=output[0]/255
            print(outimg.shape)
        cv2.imshow("input", img)
        cv2.imshow("output", outimg)
        cv2.waitKey(0)

    def test(self,imgparts):
        outImages=[]
        input = tf.placeholder(dtype=tf.float32, shape=(None,INPUT_WIDTH, INPUT_HEIGHT, 3))
        predicts = self.build_network(input)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state("train_data\model_file")
            if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            for i in range(int(len(imgparts) / groups)):
                images=imgparts[i*groups:(i+1)*groups]
                outputs = sess.run(predicts, feed_dict={input: images})
                outImages[i*groups:(i+1)*groups]=outputs
            #output=sess.run(predicts, feed_dict={input: images})
        return outImages





