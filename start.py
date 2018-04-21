import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf

import img_convert
from CDAE_hope import My_net
from functools import reduce

#import win32api,win32con

PATCH_SIZE=32
STRIDE=1
INPUT_HEIGHT=32
INPUT_WIDTH=32
NUM_CHANNEL=1
lr = 0.001
epochs = 50
num_patch = 1000


FILE_PATH = './test_data/'
FILE_NONE_DEFECTIVE = '1-0.png'
# FILE_DEFECTIVE = 'fb7_b.bmp'#''bag_d01.bmp'
FILE_DEFECTIVE = '1-1.png'
MODEL_FILE_PATH = './model/'
MODEL_NAME = 'cdaen.ckpt'

x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL), name='x-input')
y = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNEL), name='y-input')

CDAEnet = My_net()
predict = CDAEnet.buildnet(x)
loss = CDAEnet.loss_function(predict, y)
loss2 = CDAEnet.loss_function(predict, x)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
saver = tf.train.Saver()


filepath_src = FILE_PATH+FILE_NONE_DEFECTIVE
im_src = cv.imread(filepath_src, 0)
im_src = cv.resize(im_src,(256,256))
imgCvt = img_convert.ImgConvert(im_src, stride=STRIDE, patch_size=PATCH_SIZE)
patches = imgCvt.im2rnd_patches(num_patch = num_patch)

filepath_def = FILE_PATH+FILE_DEFECTIVE
im_def = cv.imread(filepath_def, 0)
im_def = cv.resize(im_def,(256,256))
imgCvt2 = img_convert.ImgConvert(im_def, stride=32, patch_size=PATCH_SIZE)
patches_src, col, row, num_patches  = imgCvt2.im2reg_patches()


def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = list(filter_maps.shape)
    reshaped_maps = filter_maps.reshape((dimension[1] * dimension[2], dimension[3]))

    # Compute the inner product to get the gram matrix
    if dimension[1] * dimension[2] > dimension[3]:
        return np.dot(reshaped_maps.T, reshaped_maps)
    else:
        return np.dot(reshaped_maps, reshaped_maps.T)

def loss_texture_feature(list1, list2):
	n = len(list1)
	gram_loss = 0
	for i in range(n):
		x_layer_maps = list1[i]
		t_layer_maps = list2[i]
		x_layer_gram = convert_to_gram(x_layer_maps)
		t_layer_gram = convert_to_gram(t_layer_maps)
		shape = list(x_layer_maps.shape)
		size = reduce(lambda a, b: a * b, shape) ** 2
		#gram_loss = numpy.sqrt(numpy.sum(numpy.square(x_layer_gram - t_layer_gram)))
		#gram_loss = np.linalg.norm(x_layer_gram - t_layer_gram)
		gram_loss = np.linalg.norm(x_layer_maps - t_layer_maps)

	return gram_loss / size




with tf.Session() as sess:

	init = tf.global_variables_initializer()
	sess.run(init)
	
	checkpoint = tf.train.get_checkpoint_state(MODEL_FILE_PATH)
	if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Loaded successfully:", checkpoint.model_checkpoint_path)

############   train   #####################
	
	# CDAEnet.train(sess, patches, optimizer, loss, x, y, epochs)

	# model_path = os.path.join(MODEL_FILE_PATH, MODEL_NAME)
	# saver.save(sess, model_path)
	# print('Saving the training model file: ' + os.path.basename(model_path))

############    test   ######################

	# recons_patches = CDAEnet.evaluate(sess, patches_src, predict, loss2, x)

	# print('evaluation done!')

	# # reconstruct
	# imout,_ = imgCvt2.reg_patches2img(recons_patches,col,row)
	# out_x = imout.shape[0]
	# out_y = imout.shape[1]
	# im_def1 = im_def[0:out_x, 0:out_y]
	# print('reconstruction done!')

	# # show results
	# plt.figure()
	# plt.subplot(1, 3, 1)
	# plt.imshow(im_def1, cmap='gray')
	# plt.title('im_test_normalized')
	# plt.subplot(1, 3, 2)
	# plt.imshow(imout, cmap='gray')
	# plt.title('reconstruction')
	# plt.subplot(1, 3, 3)
	# im_dif = abs(im_def1.astype(np.float32) - imout.astype(np.float32))
	# plt.imshow(im_dif, cmap='gray')
	# plt.title('dif')
	# plt.show()

############     feature map   ######################
	feature_sum = CDAEnet.feature_train(sess, patches, x)
	loss_list = []
	#print(feature_sum[0].shape)
	for i in range(num_patches):
		feature_map = CDAEnet.feature_test(sess, patches_src[i], x)

		loss_texture = loss_texture_feature(feature_sum, feature_map)
		loss_list.append(loss_texture)

	loss_mat = np.array(loss_list).reshape((col,row))
	plt.figure()
	plt.subplot(1, 3, 1)
	plt.imshow(im_src, cmap='gray')
	plt.subplot(1, 3, 2)
	plt.imshow(im_def, cmap='gray')
	plt.subplot(1, 3, 3)
	plt.imshow(loss_mat, cmap=plt.cm.viridis)#
	plt.show()