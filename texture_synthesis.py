import numpy as np
import scipy
import tensorflow as tf
import skimage 
import vgg19
import utils 
import time
import os
import sys
import shutil
import img_convert
import matplotlib.pyplot as plt
from functools import reduce

#img_name = str(sys.argv[1])
img_name = '1-1.png'
defect_name = '1-1.png'
img_path = './test_data/'
patch_size = 16
num_channel = 3

tex_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
tex_weights = [1e-9,1e-9,1e-9,1e-9,1e-9]

def change(img,patch_size):
    patches = []
    num_patches = 0
    size_src = img.shape  # (256,256)
    x_max = int((size_src[1] - patch_size)/patch_size + 1)
    y_max = int((size_src[0] - patch_size)/patch_size + 1)
    for y in range(y_max):#rows
        r = y * patch_size
        for x in range(x_max):#cols
            c = x * patch_size
            patch = img[r:r + patch_size, c:c + patch_size]
            num_patches += 1
            patches.append(patch)
    return patches, y_max, x_max, num_patches

img1 = utils.load_image(img_path+img_name)
img2 = utils.load_image(img_path+defect_name)

patch1, col, row, num_patches = change(img1, patch_size)
# patch_mean = sum(patch1)/num_patches
patch2,_ ,_,_= change(img2, patch_size)


x = tf.placeholder(dtype=tf.float32, shape=(1, patch_size, patch_size, num_channel), name='x-input')
y = tf.placeholder(dtype=tf.float32, shape=(1, patch_size, patch_size, num_channel), name='y-input')

def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # Compute the inner product to get the gram matrix
    if dimension[1] * dimension[2] > dimension[3]:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)
    else:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_b=True)

def get_texture_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):
        # Compute gram matrices using the activated filter maps of the art and generated images
        x_layer_maps = getattr(x, l)
        t_layer_maps = getattr(s, l)
        x_layer_gram = convert_to_gram(x_layer_maps)
        t_layer_gram = convert_to_gram(t_layer_maps)

        # Make sure the feature map dimensions are the same
        assert_equal_shapes = tf.assert_equal(x_layer_maps.get_shape(), t_layer_maps.get_shape())
        with tf.control_dependencies([assert_equal_shapes]):
            # Compute and return the normalized gram loss using the gram matrices
            shape = x_layer_maps.get_shape().as_list()
            size = reduce(lambda a, b: a * b, shape) ** 2
            gram_loss = tf.nn.l2_loss((x_layer_gram - t_layer_gram))/2
            return gram_loss / size

def get_l2_loss_for_layer(x, s, l):
    with tf.name_scope('get_l2_loss_for_layer'):
        x_layer_maps = getattr(x, l)
        t_layer_maps = getattr(s, l)
        l2_loss = tf.nn.l2_loss((x_layer_maps - t_layer_maps))/2
        return l2_loss

#####################################################
#1.
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4   
#2.
os.environ["CUDA_VISIBLE_DEVICEs"]="0"

with tf.Session(config=config) as sess:
    vgg = vgg19.Vgg19()
    vgg2 = vgg19.Vgg19()
    with tf.name_scope("origin"):
        vgg.build(x, patch_size)
    with tf.name_scope("new"):
        vgg2.build(y, patch_size)
    ## Caculate the Loss according to the paper
    loss_sum = 0.
    for i,layer in enumerate(tex_layers):
        loss = tex_weights[i] * get_texture_loss_for_layer(vgg, vgg2, layer)
        #loss = tex_weights[i] * get_l2_loss_for_layer(vgg, vgg2, layer)
        loss_sum = tf.add(loss_sum,loss)

        # origin = getattr(vgg, layer)
        # new = getattr(vgg2, layer)
        # shape = origin.get_shape().as_list()
        # N = shape[3]
        # M = shape[1]*shape[2]
        # F = tf.reshape(origin,(-1,N))
        # Gram_o = (tf.matmul(tf.transpose(F),F)/(N*M))
        # F_t = tf.reshape(new,(-1,N))
        # Gram_n = tf.matmul(tf.transpose(F_t),F_t)/(N*M)
        # loss = tex_weights[i] * tf.nn.l2_loss((Gram_o-Gram_n))/2
        #loss_sum = tf.add(loss_sum,loss)

    sess.run(tf.global_variables_initializer())

    loss_list = []

    for i in range(col):
        for j in range(row):
            index = i * row + j
            flag_u = 1
            flag_d = 1
            flag_l = 1
            flag_r = 1
            if j == 0:
                flag_u = -1
            elif j == row-1:
                flag_d = -1
            if i == 0:
                flag_l = -1
            elif i == row-1:
                flag_r = -1
            batch0 = patch2[index].reshape((1,patch_size,patch_size,num_channel)).astype("float32")
            #batch2 = patch2[i-1].reshape((1,patch_size,patch_size,num_channel)).astype("float32")
            batch_u = patch2[index - flag_u].reshape((1,patch_size,patch_size,num_channel)).astype("float32")
            batch_d = patch2[index + flag_d].reshape((1,patch_size,patch_size,num_channel)).astype("float32")
            batch_l = patch2[index - row * flag_l].reshape((1,patch_size,patch_size,num_channel)).astype("float32")
            batch_r = patch2[index + row * flag_r].reshape((1,patch_size,patch_size,num_channel)).astype("float32")
            # if i==num_patches-1:
            #     batch3 = patch2[0].reshape((1,patch_size,patch_size,num_channel)).astype("float32")
            # else:
            #     batch3 = patch2[i+1].reshape((1,patch_size,patch_size,num_channel)).astype("float32")    
            loss_u = sess.run(loss_sum, feed_dict={x: batch0, y: batch_u})
            loss_d = sess.run(loss_sum, feed_dict={x: batch0, y: batch_d})
            loss_l = sess.run(loss_sum, feed_dict={x: batch0, y: batch_l})
            loss_r = sess.run(loss_sum, feed_dict={x: batch0, y: batch_r})
            loss = min(loss_u,loss_d,loss_l,loss_r)
            # loss = sess.run(loss_sum, feed_dict={x: batch1, y: batch_mean})       
            print('loss=' + str(loss))  
            loss_list.append(loss)

    loss_mat = np.array(loss_list).reshape((col,row))
    print(loss_mat)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.subplot(1, 3, 3)
    plt.imshow(loss_mat, cmap=plt.cm.viridis)#
    plt.show()
