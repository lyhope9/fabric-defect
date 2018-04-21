import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

import img_convert
from CDAENet04 import Network,MODEL_FILE_PATH

import win32api,win32con

PATCH_SIZE=32
STRIDE=1

# FILE_PATH_DICT = {'HOME':'D:/deepLearning/data/fabric_images/forPaper',
# 				  'OFFICE':'D:/[my documents]/images/forPaper/',
# 				  'LAB':'../images/forPaper/'}

#FILE_PATH = FILE_PATH_DICT['OFFICE']
#FILE_NONE_DEFECTIVE = 'fb7_b.bmp'
FILE_PATH = './images'
FILE_NONE_DEFECTIVE = 'd.jpg'
# FILE_DEFECTIVE = 'fb7_b.bmp'#''bag_d01.bmp'
FILE_DEFECTIVE = 'd.jpg'

def per_image_standardization(im_src, zero_center = True):
    '''''stat = ImageStat.Stat(img)
    mean = stat.mean
    stddev = stat.stddev
    img = (np.array(img) - stat.mean)/stat.stddev'''
    size_src = im_src.shape
    n_channel = len(size_src)

    num_compare = size_src[0] * size_src[1] * n_channel
    img_arr = np.array(im_src)

    if zero_center: # -1~1 normalize
        im_mean = np.mean(img_arr)
        im_dev = max(np.std(img_arr), 1/num_compare)
        im_src_normalized = (img_arr - im_mean)/im_dev
    else: # 0~1 normalize
        min_val = np.min(img_arr)
        max_val = np.max(img_arr)
        im_src_normalized = (img_arr - min_val) / (max_val - min_val)

    print('@@ normalizing method: zero_center = %s' % str(zero_center))

    return im_src_normalized

def normalize_image(im_src):
    '''zero-center normalizing based on tensorflow '''
    import tensorflow as tf
    # normalize image
    size_src = im_src.shape
    shape = len(size_src)
    if(shape == 2) : # 2D image
        im_src = np.reshape(im_src, newshape=[size_src[0], size_src[1], 1])

    # Subtract off the mean and divide by the variance of the pixels.
    # Linearly scales image to have zero mean and unit norm
    std_im = tf.image.per_image_standardization(im_src)

    if(shape == 2):
        im_out = std_im[:,:,0]
    else:
        im_out = std_im
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        im_normalized = sess.run(im_out)
        # debug
        im_normalized_u8 =im_normalized.astype(np.uint8)
        cv.imwrite('./normalized.png', im_normalized_u8)
    return im_normalized

def preprocess(im_src, show_img=True):
    assert (im_src.any() != None)
    if len(im_src.shape) == 3:
        im_src = cv.cvtColor(im_src, cv.CV_BRG2GRAY)

    im_src = cv.GaussianBlur(im_src, (5, 5), 0)# 0是指根据窗口大小（5,5）来计算高斯函数标准差

    # im_src_normalized = normalize_image(im_src)
    im_src_normalized = per_image_standardization(im_src, zero_center=False)
    # print('[training image] mean:%s, dev:%s' % (im_mean, im_dev))

    # display
    if show_img:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im_src, cmap='gray')
        plt.title('origin')
        plt.subplot(1, 2, 2)
        plt.imshow(im_src_normalized, cmap='gray')
        plt.title('normalized')

    return im_src_normalized

def training():

    #filepath_src = os.path.join(FILE_PATH,FILE_NONE_DEFECTIVE)
    filepath_src = FILE_PATH+'/'+FILE_NONE_DEFECTIVE
    im_src = cv.imread(filepath_src, 0)
    im_src_normalized = preprocess(im_src, show_img=True)

    cv.imshow('orgin', im_src)
    cv.waitKey(0)

    # extracting patches
    imgCvt = img_convert.ImgConvert(im_src_normalized, patch_size=PATCH_SIZE, stride=STRIDE)
    patches = imgCvt.im2rnd_patches(num_patch = 5000)

    # training
    net = Network(lr = 0.001, epochs = 1000, noise_dev = np.std(im_src_normalized))
    net.train(patches)

def detecting():
    # read image
    filepath_test = os.path.join(FILE_PATH,FILE_DEFECTIVE)
    im_test = cv.imread(filepath_test, cv.IMREAD_GRAYSCALE)

    # preprocess
    im_test_normalized = preprocess(im_test)

    # extract patches
    imgCvt = img_convert.ImgConvert(im_test_normalized, patch_size=PATCH_SIZE, stride=STRIDE)
    patches_src, col, row, num_patches  = imgCvt.im2reg_patches()

    # evaluate
    net = Network(visualizing=True)
    recons_patches = net.evaluate(patches_src)
    print('evaluation done!')

    # reconstruct
    imout,_ = imgCvt.reg_patches2img(recons_patches,col,row)
    print('reconstruction done!')

    # show results
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im_test_normalized, cmap='gray')
    plt.title('im_test_normalized')
    plt.subplot(1, 3, 2)
    plt.imshow(imout, cmap='gray')
    plt.title('reconstruction')
    plt.subplot(1, 3, 3)
    im_dif = abs(im_test_normalized.astype(np.float32) - imout.astype(np.float32))
    plt.imshow(im_dif, cmap='gray')
    plt.title('dif')
    plt.show()

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i) #取文件绝对路径
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
def cleanup():
    path = MODEL_FILE_PATH
    del_file(path)
def edge_detect(img):
    import cv2
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()
def test():
    import tensorflow as tf
    shape = [3, 4, 5]
    a = tf.Variable(tf.random_normal(shape))  # a：activations
    axis = list(range(len(shape) - 1))  # len(x.get_shape())
    a_mean, a_var = tf.nn.moments(a, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ar = sess.run( a)
        am = sess.run( a_mean)
        av = sess.run(a_var)
        print("a:", ar,'\nmean: ', am, '\nvariance', av)
        print('mean shape:', am.shape)  # (64, )
        print('variance:', av.shape)  # (64, )
        print('r:', (ar[0][0]+ar[0][1]+ar[0][2]+ar[0][3])/4.)
def test2():
    import tensorflow as tf


    x3 = tf.constant(1.0, shape=[1, 5, 5, 1])
    kernel = tf.constant(1.0, shape=[3, 3, 3, 1])
    y3 = tf.nn.conv2d_transpose(x3, kernel, output_shape=[1, 20, 18, 3], strides=[1, 4, 4, 1], padding="SAME")

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    y3_val = sess.run([ y3])

    print(y3_val.shape)

    sess.close()

if __name__ == '__main__':
    ret = win32api.MessageBox(0, "yes-开始训练; no-开始检测", "开始训练？", win32con.MB_YESNOCANCEL)
    if (ret == win32con.IDYES):
        if (win32api.MessageBox(0, "删除现有权值数据？", "确认", win32con.MB_YESNO) == win32con.IDYES):
            cleanup()
        training()
    elif (ret == win32con.IDNO):
        detecting()

    else:
        print('User canceled.')
        test2()
