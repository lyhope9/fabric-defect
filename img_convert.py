import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

class ImgConvert:
    'convert image to patches, or patches to image'
    def __init__(self, im, stride, patch_size = 32):
        self.imSrc = im
        self.patch_size = patch_size
        self.stride = stride
    def show(self):
        if min(self.imSrc.shape) != 0:
            #cv.imshow('original image', self.imSrc)
            plt.imshow(self.imSrc,cmap = 'gray')
            #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        else:
            print('No source image assigned.')
        return
    def im2rnd_patches(self, num_patch = 10000): # random patches
        patches = []
        size_src = self.imSrc.shape #(256,256)
        x_max = size_src[1] - self.patch_size + 1
        y_max = size_src[0] - self.patch_size + 1
        for i in range(num_patch):
            p_x = np.random.random_integers(low=0,high=x_max-1) # int between `low` and `high`, inclusive.
            p_y = np.random.random_integers(low=0,high=y_max-1)
            patch = self.imSrc[p_x:p_x+self.patch_size, p_y:p_y+self.patch_size]
            assert(patch.shape == (32,32))
            patches.append(patch)
        return patches

    def im2reg_patches(self): # regular patches
        # 将一幅图像规则化地分割成若干子图块并输出为list of patches
        patches = []
        num_patches = 0
        size_src = self.imSrc.shape  # (256,256)
        x_max = int((size_src[1] - self.patch_size)/self.stride + 1)
        y_max = int((size_src[0] - self.patch_size)/self.stride + 1)
        for y in range(y_max):#rows
            r = y * self.stride
            for x in range(x_max):#cols
                c = x * self.stride
                patch = self.imSrc[r:r + self.patch_size, c:c + self.patch_size]
                num_patches += 1
                patches.append(patch)
        return patches,y_max,x_max,num_patches

    def reg_patches2img(self,patches, n_row, n_col):
        # 将子图块list重新拼成完整图像并输出
        size_imout = ((n_row - 1) * self.stride + self.patch_size, (n_col - 1) * self.stride + self.patch_size)
        imout = np.zeros(size_imout,np.float32)
        for y in range(n_row):#rows
            r = y * self.stride
            for x in range(n_col):#cols
                patch = patches[ y * n_row + x]
                c = x * self.stride
                imout[r:r+self.patch_size, c:c+self.patch_size ] = patch

        # max_val = imout.max()
        # min_val = imout.min()
        #imout = imout / max_val * 255.
        # imout = imout.astype(np.uint8)
        return imout,size_imout

def test():
    # ImgConvert模块测试
    # 输入一幅图像，将其分割成若干子图块，然后重新拼接，观察原图与重建结果的差异

    FILE_PATH_DICT = {'HOME': 'D:/deepLearning/data/fabric_images/images/',
                      'OFFICE': 'D:/[my documents]/images/forPaper/',
                      'LAB': 'E:/HuGuanghua/images/',
                      'hope':'./images/'}
    FILE_PATH = FILE_PATH_DICT['hope']
    FILE_NONE_DEFECTIVE = 'd.jpg'

    filepath_src = os.path.join(FILE_PATH, FILE_NONE_DEFECTIVE)
    im_src = cv.imread(filepath_src, 0)


    imgCvt = ImgConvert(im_src, patch_size=32, stride=16)
    p, col, row, num_patches = imgCvt.im2reg_patches()
    plt.imshow(p[1], cmap='gray')
    plt.title('patch[1]')
    plt.show()


    imout, size_imout = imgCvt.reg_patches2img(p, col, row)
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(im_src, cmap='gray')
    plt.title('src')
    #
    plt.subplot(1, 4, 2)
    plt.imshow(imout, cmap='gray')
    plt.title('reconstruction')
    #imgCvt.show()
    #
    plt.subplot(1, 4, 3)
    plt.imshow(im_src - imout, cmap='gray')
    plt.title('dif')

    noise = im_src + 0.1 * np.random.randn(*im_src.shape) * 255
    plt.subplot(1, 4, 4)
    plt.imshow(noise, cmap='gray')
    plt.title('noise')

    plt.show()
    print('done.')

    return

if __name__ == '__main__':
    test()