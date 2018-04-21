import numpy as np
import tensorflow as tf
import os
import cv2
import img_convert as ic

INPUT_HEIGHT = 32
INPUT_WIDTH = 32
batch_size = 64
NUM_CHANNEL = 1
noise_factor = 0.01  ## (0~1)

MODEL_FILE_PATH = "./train_data/model_file"
MODEL_NAME = 'cdaen.ckpt'

class Network():
    def __init__(self, lr = 0.0001, epochs = 100):
        self.learning_rate = lr
        self.epochs = epochs

    def inference(self,input_tensor):
        variable_batch_size = tf.shape(input_tensor)[0]
        # layer1
        # 原始输入是32×32*1
        conv1_1 = tf.layers.conv2d(inputs=input_tensor, filters=64, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu)
        ## 池化输出16*16*64
        pool1_2 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], strides=[2,2])

        # layer2
        # 输入16*16*64
        conv2_1 = tf.layers.conv2d(inputs=pool1_2, filters=64, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu)
        # 输出8×8×64
        pool2_2 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=[2,2])

        # layer 3
        # 输入8×8×64
        conv3_1 = tf.layers.conv2d(inputs=pool2_2, filters=32, kernel_size=[3, 3], padding='same',
                                   activation=tf.nn.relu)
        # 输出4×4×32
        pool3_2 = tf.layers.max_pooling2d(inputs=conv3_1, pool_size=[2, 2], strides=[2,2])

        # deconv layer
        # 输出8×8×32
        deconv3_1 = tf.layers.conv2d_transpose(inputs=pool3_2, filters=32, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')
        # 输出16×16×64
        deconv2_1 = tf.layers.conv2d_transpose(inputs=deconv3_1, filters=64, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')
        # 输出32×32×64
        deconv1_1 = tf.layers.conv2d_transpose(inputs=deconv2_1, filters=64, kernel_size=[3, 3], strides=(2, 2),
                                               padding='same')

        # conv layer :
        # 输出32×32×1
        final = tf.layers.conv2d(inputs=deconv1_1, filters=1, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu)

        ##
        # weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, NUM_CHANNEL, 64], stddev=0.1, name='weight_1'))
        # bias_1 = tf.Variable(tf.constant(0.0, shape=[64], name='bias_1'))
        # weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1, name='weight_2'))
        # bias_2 = tf.Variable(tf.constant(0.0, shape=[64], name='bias_2'))
        # weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 32], stddev=0.1, name='weight_3'))
        # bias_3 = tf.Variable(tf.constant(0.0, shape=[32], name='bias_3'))
        #
        # deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.1), name='deconv_weight_1')
        # deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 32], stddev=0.1), name='deconv_weight_2')#注意64，32顺序
        # deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name='deconv_weight_3')
        #
        # conv1 = tf.nn.conv2d(input=input_tensor, filter=weight_1, strides=[1, 1, 1, 1], padding='SAME')
        # conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')
        # acti1 = tf.nn.relu(conv1, name='acti_1')
        # pool1 = tf.nn.max_pool(value=acti1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
        #
        # conv2 = tf.nn.conv2d(input=pool1, filter=weight_2, strides=[1, 1, 1, 1], padding='SAME')
        # conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')
        # acti2 = tf.nn.relu(conv2, name='acti_2')
        # pool2 = tf.nn.max_pool(value=acti2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')
        #
        # conv3 = tf.nn.conv2d(input=pool2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME')
        # conv3 = tf.nn.bias_add(conv3, bias_3, name='conv_3')
        # acti3 = tf.nn.relu(conv3, name='acti_3')
        # pool3 = tf.nn.max_pool(value=acti3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_3')
        #
        # output_shape = tf.stack([variable_batch_size, 8, 8, 32])
        # deconv1 = tf.nn.conv2d_transpose(value=pool3, filter=deconv_weight_1, output_shape=output_shape,
        #                                  strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')
        #
        # output_shape = tf.stack([variable_batch_size, 16, 16, 64])
        # deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=output_shape,
        #                                  strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')
        #
        # output_shape = tf.stack([variable_batch_size, 32, 32, 64])
        # deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=output_shape,
        #                                  strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')
        # # deconv1 = tf.nn.conv2d_transpose(value=pool3, filter=deconv_weight_1, output_shape=[batch_size, 8, 8, 32],
        # #                                  strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')
        # # deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=[batch_size, 16, 16, 64],
        # #                                  strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')
        # # deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=[batch_size, 32, 32, 64],
        # #                                  strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')
        #
        # weight_final = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, NUM_CHANNEL], stddev=0.1, name='weight_final'))
        # bias_final = tf.Variable(tf.constant(0.0, shape=[NUM_CHANNEL], name='bias_final'))
        # conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')
        # final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')
        #
        # 返回前向传播结果
        return final

    def inference2(self, input_tensor):
        variable_batch_size = tf.shape(input_tensor)[0]

        with tf.variable_scope('inference', reuse=False):
            weight_1 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       shape=[3, 3, NUM_CHANNEL, 64], name='weight_1')
            bias_1 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[64], name='bias_1')
            weight_2 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1), shape=[3, 3, 64, 64],
                                       name='weight_2')
            bias_2 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[64], name='bias_2')
            weight_3 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1), shape=[3, 3, 64, 32],
                                       name='weight_3')
            bias_3 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[32], name='bias_3')

            deconv_weight_1 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 3, 32, 32], name='deconv_weight_1')
            deconv_weight_2 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 3, 64, 32], name='deconv_weight_2')  # 注意64，32顺序
            deconv_weight_3 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 3, 64, 64], name='deconv_weight_3')

            conv1 = tf.nn.conv2d(input=input_tensor, filter=weight_1, strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')
            acti1 = tf.nn.relu(conv1, name='acti_1')
            pool1 = tf.nn.max_pool(value=acti1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                   name='max_pool_1')

            conv2 = tf.nn.conv2d(input=pool1, filter=weight_2, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')
            acti2 = tf.nn.relu(conv2, name='acti_2')
            pool2 = tf.nn.max_pool(value=acti2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                   name='max_pool_2')

            conv3 = tf.nn.conv2d(input=pool2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.bias_add(conv3, bias_3, name='conv_3')
            acti3 = tf.nn.relu(conv3, name='acti_3')
            pool3 = tf.nn.max_pool(value=acti3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                   name='max_pool_3')

            output_shape = tf.stack([variable_batch_size, 8, 8, 32])
            deconv1 = tf.nn.conv2d_transpose(value=pool3, filter=deconv_weight_1, output_shape=output_shape,
                                             strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')

            output_shape = tf.stack([variable_batch_size, 16, 16, 64])
            deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=output_shape,
                                             strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')

            output_shape = tf.stack([variable_batch_size, 32, 32, 64])
            deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=output_shape,
                                             strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')

            weight_final = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           shape=[3, 3, 64, NUM_CHANNEL], name='weight_final')
            bias_final = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[NUM_CHANNEL],
                                         name='bias_final')
            conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')
            final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')
        #
        # 返回前向传播结果
        return final
    def loss_function(self,predicts,labels):
        loss = tf.reduce_mean(tf.pow(tf.subtract(predicts, labels), 2.0))  #此处tf.pow为幂运算函数
        return loss

    def patch_list2matrix(self,batch_list):# batches is list
        n = len(batch_list)
        batches = np.zeros(dtype=np.float32, shape=(n,INPUT_HEIGHT, INPUT_WIDTH))
        for i in range(n):
            batches[i,:,:] = batch_list[i]
        training_data = np.reshape(batches, newshape=[n, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL])
        return training_data

    def train(self,training_data):
        input = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL), name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL), name='y-input')
        predicts = self.inference(input)
        global_step = tf.Variable(0, trainable=False) # 每个batch增1
        loss = self.loss_function(predicts, y_)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,global_step=global_step)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Training start......')
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state(MODEL_FILE_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Loaded successfully:", checkpoint.model_checkpoint_path)

            # reshape data
            n = len(training_data)
            training_data = self.patch_list2matrix(training_data)

            for j in range(self.epochs+1):
                np.random.shuffle(training_data)
                mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
                for mini_batch in mini_batches:
                    noise_x = mini_batch + noise_factor * np.random.randn(*mini_batch.shape) * 255
                    noise_x = np.clip(noise_x, 0., 255.)
                    _,loss_val,step = sess.run([train_step, loss, global_step], feed_dict={input: noise_x, y_: mini_batch})
                print("Epoch {0} complete".format(j))

                if j % 10 == 0 or j == self.epochs:
                    print('After %d training steps, loss on training batch is: %g ' % (step, loss_val))
                    model_path = os.path.join(MODEL_FILE_PATH, MODEL_NAME)
                    saver.save(sess, model_path, global_step=step)
                    print('Saving the training model file: ' + os.path.basename(model_path))

    def evaluate(self, patches):
        input = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL))
        predicts = self.inference(input)
        loss = self.loss_function(predicts, input)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state(MODEL_FILE_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Model loaded successfully:", checkpoint.model_checkpoint_path)

            # reshape data
            n = len(patches)
            testing_data = self.patch_list2matrix(patches)
            reconstructed_batches = []
            mini_batches = [testing_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                recontructed_batch, loss_val = sess.run([predicts,loss], feed_dict={input: mini_batch}) # recontructed_batch 是4维矩阵 ： batch_size * H * W * 1
                reconstructed_batches.append(recontructed_batch)
            reconst_patches = self.tensorlist2patches(reconstructed_batches)
        return reconst_patches
    def tensorlist2patches(self, batches):
        patches = []
        n = len(batches)
        for i in range(n):
            num_im = batches[i].shape[0] # how many images are there in current batch
            for j in range(num_im):
                patch = np.reshape(batches[i][j], newshape=(INPUT_HEIGHT, INPUT_WIDTH))
                patches.append(patch )
        return patches





