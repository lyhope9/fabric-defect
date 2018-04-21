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
from CDAE_hope import My_net

#img_name = str(sys.argv[1])
img_name = '1-1.png'
noise_name = '1-1.png'
img_path = './test_data/'
MODEL_FILE_PATH = './model_texture/'
MODEL_NAME = 'cdaen.ckpt'
patch_size = 32
epochs = 300
num_patch = 500

tex_layers = ['pool3', 'pool2', 'pool1', 'conv1_1']
tex_weights = [1e-9,1e-9,1e-9,1e-9]

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

def im2rnd_patches(img, num_patch = 10000): # random patches
        patches = []
        size_src = img.shape #(256,256)
        x_max = size_src[1] - 32 + 1
        y_max = size_src[0] - 32 + 1
        for i in range(num_patch):
            p_x = np.random.random_integers(low=0,high=x_max-1) # int between `low` and `high`, inclusive.
            p_y = np.random.random_integers(low=0,high=y_max-1)
            patch = img[p_x:p_x+32, p_y:p_y+32]
            assert(patch.shape == (32,32,3))
            patches.append(patch)
        return patches



img1 = utils.load_image(img_path+img_name)
img2 = utils.load_image(img_path+noise_name)
patch1, col, row, num_patches = change(img1, 32)
patch2,_ ,_,_= change(img2, 32)
patches_train = im2rnd_patches(img1, num_patch)


x = tf.placeholder(dtype=tf.float32, shape=(1, 32, 32, 3), name='x-input')
y = tf.placeholder(dtype=tf.float32, shape=(1, 32, 32, 3), name='y-input')
z = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='z-input')
w = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='w-input')

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


##############################################
cdae_x = My_net()
predict_x = cdae_x.buildnet(x)
cdae_y = My_net()
predict_y = cdae_y.buildnet(y)
cdae_z = My_net()
predict_z = cdae_z.buildnet(z)

loss_z = cdae_z.loss_function(predict_z, w)

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss_z)
saver = tf.train.Saver()


## Caculate the Loss according to the paper
loss_sum = 0.
for i,layer in enumerate(tex_layers):
    loss = tex_weights[i] * get_texture_loss_for_layer(cdae_x, cdae_y, layer)
    loss_sum = tf.add(loss_sum,loss)
#####################################################
#1.
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4   
#2.
os.environ["CUDA_VISIBLE_DEVICEs"]="0"

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(MODEL_FILE_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:  # 加载训练好的网络数据
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded successfully:", checkpoint.model_checkpoint_path)

    cdae_z.train(sess, patches_train, optimizer, loss_z, z, w, epochs)

    model_path = os.path.join(MODEL_FILE_PATH, MODEL_NAME)
    saver.save(sess, model_path)
    print('Saving the training model file: ' + os.path.basename(model_path))

    loss_list = []

    for i in range(num_patches):
        batch1 = patch1[i].reshape((1,32,32,3)).astype("float32")
        batch2 = patch1[i].reshape((1,32,32,3)).astype("float32")
        loss = sess.run(loss_sum, feed_dict={x: batch1, y: batch2})
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
