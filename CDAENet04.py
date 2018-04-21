import numpy as np
import tensorflow as tf
import os
import net_visualize
import tensorflow.contrib as contrib
import cv2 as cv

# global parameters
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
batch_size = 50
NUM_CHANNEL = 1
noise_factor = 0.001  ## (0~1)
NUM_FILTER = 32
lambd = 0.0001 ## (0~1)

CLASS_VERSION = '_04'
MODEL_FILE_PATH = "./model" + CLASS_VERSION  # 模型保存路径
MODEL_NAME = 'CDAENet%s.ckpt' % CLASS_VERSION  # 模型文件名
WEIGHT_FOLDER = './weights' + CLASS_VERSION
FEATURE_FOLDER = './feature_maps' + CLASS_VERSION


class Network():
    def __init__(self, lr=0.001, epochs=100, noise_dev=255, visualizing=True):
        '''If visualizing is Ture, then the network will show weights and feature maps'''
        self.learning_rate = lr
        self.epochs = epochs
        self.noise_dev = noise_dev
        self.infer_func = self.inference_5by5
        self.loss_func = self.loss_function_corr
        self.visualizing = visualizing
        self.create_folder(MODEL_FILE_PATH)

    def create_folder(self, filepath):
        '''创建模型保存用的文件夹'''
        # 判断路径是否存在
        isExists = os.path.exists(filepath)

        if not isExists:
            # 如果不存在则创建目录
            os.makedirs(filepath)
            print(filepath + ' 创建成功')
        return True

    def inference_sigmoid(self, input_tensor):
        with tf.variable_scope('inference', reuse=False):
            variable_batch_size = tf.shape(input_tensor)[0]

            # conv_layer #1
            weight_1 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       shape=[5, 5, NUM_CHANNEL, NUM_FILTER], name='weight_1')
            bias_1 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[int(NUM_FILTER)], name='bias_1')
            conv_1 = tf.nn.conv2d(input=input_tensor, filter=weight_1, strides=[1, 1, 1, 1], padding='SAME',
                                  name='conv_1')
            ae_1 = tf.nn.bias_add(conv_1, bias_1, name='ae_1')
            acti_1 = tf.nn.sigmoid(ae_1, name='acti_1')
            pool_1 = tf.nn.max_pool(value=ae_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='pool_1')

            # conv_layer #2
            weight_2 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       shape=[3, 3, NUM_FILTER, NUM_FILTER], name='weight_2')
            bias_2 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[NUM_FILTER], name='bias_2')
            conv_2 = tf.nn.conv2d(input=pool_1, filter=weight_2, strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            ae_2 = tf.nn.bias_add(conv_2, bias_2, name='ae_2')
            acti_2 = tf.nn.sigmoid(ae_2, name='acti_2')
            pool_2 = tf.nn.max_pool(value=ae_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='pool_2')

            # conv_layer #3
            weight_3 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       shape=[3, 3, int(NUM_FILTER), int(NUM_FILTER // 2)], name='weight_3')
            bias_3 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[int(NUM_FILTER // 2)],
                                     name='bias_3')
            conv_3 = tf.nn.conv2d(input=pool_2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            ae_3 = tf.nn.bias_add(conv_3, bias_3, name='ae_3')
            acti_3 = tf.nn.sigmoid(ae_3, name='acti_3')
            pool_3 = tf.nn.max_pool(value=ae_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='pool_3')  # 4*4*32

            # deconv_layer #1
            deconv_weight_1 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 3, int(NUM_FILTER // 2), int(NUM_FILTER // 2)],
                                              name='deconv_weight_1')
            output_shape_1 = tf.stack([variable_batch_size, 8, 8, int(NUM_FILTER // 2)])
            deconv_1 = tf.nn.conv2d_transpose(value=pool_3, filter=deconv_weight_1, output_shape=output_shape_1,
                                              strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')  # 8*8*32
            # deconv_layer #2
            deconv_weight_2 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 3, int(NUM_FILTER), int(NUM_FILTER // 2)],
                                              name='deconv_weight_2')  # 注意64，32顺序
            output_shape_2 = tf.stack([variable_batch_size, 16, 16, int(NUM_FILTER)])
            deconv_2 = tf.nn.conv2d_transpose(value=deconv_1, filter=deconv_weight_2, output_shape=output_shape_2,
                                              strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')  # 16*16*64

            # deconv_layer #3
            deconv_weight_3 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[5, 5, int(NUM_FILTER), int(NUM_FILTER)],
                                              name='deconv_weight_3')
            output_shape_3 = tf.stack([variable_batch_size, 32, 32, int(NUM_FILTER)])
            deconv_3 = tf.nn.conv2d_transpose(value=deconv_2, filter=deconv_weight_3, output_shape=output_shape_3,
                                              strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')

            # output: conv_layer #final
            weight_final = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           shape=[3, 3, int(NUM_FILTER), NUM_CHANNEL], name='weight_final')
            bias_final = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[NUM_CHANNEL],
                                         name='bias_final')
            conv_final = tf.nn.conv2d(input=deconv_3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME',
                                      name='conv_final')
            ae_final = tf.nn.bias_add(conv_final, bias_final, name='ae_final')
        #
        # 返回前向传播结果
        return ae_final, [pool_1, pool_2, pool_3, deconv_1, deconv_2, deconv_3]

    def inference_5by5(self, input_tensor):
        with tf.variable_scope('inference', reuse=False):
            variable_batch_size = tf.shape(input_tensor)[0]

            # conv_layer #1
            weight_1 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       shape=[5, 5, NUM_CHANNEL, NUM_FILTER], name='weight_1')
            tf.add_to_collection('losses', contrib.layers.l2_regularizer(lambd)(weight_1))
            bias_1 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[int(NUM_FILTER)], name='bias_1')
            conv_1 = tf.nn.conv2d(input=input_tensor, filter=weight_1, strides=[1, 1, 1, 1], padding='SAME',
                                  name='conv_1')
            ae_1 = tf.nn.bias_add(conv_1, bias_1, name='ae_1')
            acti_1 = tf.nn.relu(ae_1, name='acti_1')
            pool_1 = tf.nn.max_pool(value=acti_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='pool_1')

            # conv_layer #2
            weight_2 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       shape=[3, 3, NUM_FILTER, NUM_FILTER], name='weight_2')
            tf.add_to_collection('losses', contrib.layers.l2_regularizer(lambd)(weight_2))
            bias_2 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[NUM_FILTER], name='bias_2')
            conv_2 = tf.nn.conv2d(input=pool_1, filter=weight_2, strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            ae_2 = tf.nn.bias_add(conv_2, bias_2, name='ae_2')
            acti_2 = tf.nn.relu(ae_2, name='acti_2')
            pool_2 = tf.nn.max_pool(value=acti_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='pool_2')

            # conv_layer #3
            weight_3 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       shape=[3, 3, int(NUM_FILTER), int(NUM_FILTER // 2)], name='weight_3')
            tf.add_to_collection('losses', contrib.layers.l2_regularizer(lambd)(weight_3))
            bias_3 = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[int(NUM_FILTER // 2)],
                                     name='bias_3')
            conv_3 = tf.nn.conv2d(input=pool_2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            ae_3 = tf.nn.bias_add(conv_3, bias_3, name='ae_3')
            acti_3 = tf.nn.relu(ae_3, name='acti_3')
            pool_3 = tf.nn.max_pool(value=acti_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name='pool_3')  # 4*4*32

            # deconv_layer #1
            deconv_weight_1 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 3, int(NUM_FILTER // 2), int(NUM_FILTER // 2)],
                                              name='deconv_weight_1')
            tf.add_to_collection('losses', contrib.layers.l2_regularizer(lambd)(deconv_weight_1))
            output_shape_1 = tf.stack([variable_batch_size, 8, 8, int(NUM_FILTER // 2)])
            deconv_1 = tf.nn.conv2d_transpose(value=pool_3, filter=deconv_weight_1, output_shape=output_shape_1,
                                              strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')  # 8*8*32
            # deconv_layer #2
            deconv_weight_2 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 3, int(NUM_FILTER), int(NUM_FILTER // 2)],
                                              name='deconv_weight_2')  # 注意64，32顺序
            tf.add_to_collection('losses', contrib.layers.l2_regularizer(lambd)(deconv_weight_2))
            output_shape_2 = tf.stack([variable_batch_size, 16, 16, int(NUM_FILTER)])
            deconv_2 = tf.nn.conv2d_transpose(value=deconv_1, filter=deconv_weight_2, output_shape=output_shape_2,
                                              strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')  # 16*16*64

            # deconv_layer #3
            deconv_weight_3 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[5, 5, int(NUM_FILTER), int(NUM_FILTER)],
                                              name='deconv_weight_3')
            tf.add_to_collection('losses', contrib.layers.l2_regularizer(lambd)(deconv_weight_3))
            output_shape_3 = tf.stack([variable_batch_size, 32, 32, int(NUM_FILTER)])
            deconv_3 = tf.nn.conv2d_transpose(value=deconv_2, filter=deconv_weight_3, output_shape=output_shape_3,
                                              strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')

            # output: conv_layer #final
            weight_final = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           shape=[3, 3, int(NUM_FILTER), NUM_CHANNEL], name='weight_final')
            tf.add_to_collection('losses', contrib.layers.l2_regularizer(lambd)(weight_final))
            bias_final = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[NUM_CHANNEL],
                                         name='bias_final')
            conv_final = tf.nn.conv2d(input=deconv_3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME',
                                      name='conv_final')
            ae_final = tf.nn.bias_add(conv_final, bias_final, name='ae_final')
        #
        # 返回前向传播结果
        return ae_final, [pool_1, pool_2, pool_3, deconv_1, deconv_2, deconv_3]

    def loss_function(self, predicts, labels):
        loss = tf.reduce_mean(tf.pow(tf.subtract(predicts, labels), 2.0))
        return loss

    def loss_function_corr(self, predicts, labels):
        '''考虑两幅图像的相关系数
        a = a - mean2(a);
        b = b - mean2(b);
        r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));
        '''
        shape = predicts.get_shape()
        axis = list(range(1, len(shape)))  # 1,2,3, no 0
        corr = tf.reduce_sum(tf.multiply(predicts, labels), axis) \
               / tf.sqrt(tf.reduce_sum(tf.pow(predicts, 2), axis)
                       * tf.reduce_sum(tf.pow(labels, 2), axis))
        loss_corr = 1.0 / tf.reduce_mean(corr) # inverse of correlation
        loss_mse = tf.reduce_mean(tf.pow(tf.subtract(predicts, labels), 2.0)) # mean squared error

        # tf.add_to_collection('losses', loss_corr)
        tf.add_to_collection('losses', loss_mse)
        loss = tf.add_n(tf.get_collection('losses'))

        return loss

    def loss_function2(self, predicts, labels):
        shape = predicts.get_shape()
        axis = list(range(1, len(shape)))  # 1,2,3, no 0
        mean, var = tf.nn.moments(predicts, axis)
        mean2, var2 = tf.nn.moments(labels, axis)
        d_m = tf.reduce_mean(tf.pow(tf.subtract(mean, mean2), 2.0))
        d_m2 = tf.reduce_mean(tf.pow(mean, 2.0))
        d_v = tf.reduce_mean(tf.pow(var, 2.0))
        loss2 = d_v  # / d_m2
        loss = d_m * 10 + loss2
        # loss = tf.reduce_mean(tf.pow(tf.subtract(predicts, labels), 2.0)) + loss2
        return loss

    def patch_list2matrix(self, batch_list):  # batches is list
        n = len(batch_list)
        batches = np.zeros(dtype=np.float32, shape=(n, INPUT_HEIGHT, INPUT_WIDTH))
        for i in range(n):
            batches[i, :, :] = batch_list[i]
        training_data = np.reshape(batches, newshape=[n, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL])
        return training_data

    def train(self, image_list_for_training):
        input = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL), name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL), name='y-input')
        predicts, _ = self.infer_func(input)
        global_step = tf.Variable(0, trainable=False)  # 每个batch增1
        loss = self.loss_func(predicts, y_)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Training start......')
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state(MODEL_FILE_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Loaded successfully:", checkpoint.model_checkpoint_path)

            # reshape data
            n = len(image_list_for_training)
            patch_tensor = self.patch_list2matrix(image_list_for_training)

            for j in range(self.epochs + 1):
                np.random.shuffle(patch_tensor)
                mini_batches = [patch_tensor[k:k + batch_size] for k in range(0, n, batch_size)]
                for mini_batch in mini_batches:
                    noise_x = mini_batch + noise_factor * np.random.randn(*mini_batch.shape) * self.noise_dev
                    noise_x = np.clip(noise_x, 0., 255.)
                    _, loss_val, step = sess.run([train_step, loss, global_step],
                                                 feed_dict={input: noise_x, y_: mini_batch})
                print("Epoch {0} complete".format(j))
                print('After %d training steps, loss on training batch is: %g ' % (step, loss_val))

                if j % 10 == 0 or j == self.epochs:
                    # print('After %d training steps, loss on training batch is: %g ' % (step, loss_val))
                    model_path = os.path.join(MODEL_FILE_PATH, MODEL_NAME)
                    saver.save(sess, model_path, global_step=step)
                    print('Saving the training model file: ' + os.path.basename(model_path))

    def evaluate(self, patches):
        input = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL))
        predicts, net_state = self.infer_func(input)
        loss = self.loss_func(predicts, input)
        saver = tf.train.Saver()

        # visualize_layers = ['max_pool_1', 'max_pool_2', 'max_pool_3']

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
                recontructed_batch, loss_val = sess.run([predicts, loss], feed_dict={
                    input: mini_batch})  # recontructed_batch 是4维矩阵 ： batch_size * H * W * 1
                reconstructed_batches.append(recontructed_batch)

            reconst_patches = self.tensorlist2patches(reconstructed_batches)

            if (self.visualizing == True):
                # visualizing weights
                self.show_weights()
                # visualizing features
                feature_maps = sess.run(net_state, feed_dict={
                    input: mini_batches[-1]})  # the last batch is used to generate feature map
                self.show_feature_maps(feature_maps)

        return reconst_patches

    def show_weights(self, sess=None):
        weight_folder_name = WEIGHT_FOLDER
        self.create_folder(weight_folder_name)
        # visualizing weights
        with tf.variable_scope('inference', reuse=True):
            weight_names = ['weight_1', 'weight_2', 'weight_3', 'deconv_weight_1', 'deconv_weight_2', 'deconv_weight_3']
            for j in range(len(weight_names)):
                print('plot %s' % weight_names[j])
                conv_weights = tf.get_variable(weight_names[j]).eval()
                net_visualize.plot_conv_weights(conv_weights, weight_folder_name, weight_names[j],
                                                channels_all=False, filters_all=True) #channels_all、filters_all同时为true将产生大量子图，如32*32

            # conv_weights=[]
            # for j in range(len(weight_names)):
            #     conv_weights.append (tf.get_variable(weight_names[j]))
            # conv_weights_val = sess.run(conv_weights)
            # for j in range(len(weight_names)):
            #     net_visualize.plot_conv_weights(conv_weights_val[j], weight_folder_name, weight_names[j], filters_all=True)

    def show_feature_maps(self, feature_maps):
        '''visualizing feature maps'''
        feature_folder_name = FEATURE_FOLDER
        self.create_folder(feature_folder_name)
        for j in range(len(feature_maps)):
            print('plot feature map %s' % (j + 1))
            feature_map_name = 'feature_map_' + str(j + 1)
            net_visualize.plot_conv_output(feature_maps[j], feature_folder_name, feature_map_name, filters_all=True)



    def tensorlist2patches(self, batches):
        patches = []
        n = len(batches)
        for i in range(n):
            num_im = batches[i].shape[0]  # how many images are there in current batch
            for j in range(num_im):
                patch = np.reshape(batches[i][j], newshape=(INPUT_HEIGHT, INPUT_WIDTH))
                patches.append(patch)
        return patches





