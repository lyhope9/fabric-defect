import cv2
import numpy as np
import pickle
import random
import os
from AENetwork import Network


Width=32
Height=32
num_samples=5000


def create_data():
    all_data = []
    names = open('images/imagename.txt').read().strip().split()
    data_file='images/all_data.pkl'

    if os.path.exists(data_file):
        with open(data_file,'rb') as f_data:
            all_data=pickle.load(f_data)
        return all_data
    else:
        f_data=open(data_file,'wb')

    for name in names:
        namepath = 'images/' + name
        img = cv2.imread(namepath)
        rows,cols,_=img.shape

        #image = np.transpose(img, (1, 0, 2))   #对于图片而言竖直向下为x轴，水平向右为y轴，与张量不同，故需要转换维度
        for i in range(num_samples):
            tlx=random.randint(0,cols-Width)
            tly=random.randint(0,rows-Height)

            image=img[tlx:tlx+Width,tly:tly+Height]
            all_data.append(image)
            print(image.shape)

    pickle.dump(all_data, f_data)
    f_data.close()

    return all_data

def ImgDiv(img):
    rows, cols, _ = img.shape
    imgparts=[]
    for i in range(rows-Height):
        for j in range(cols-Width):
            imgp=img[j:j+Width,i:i+Height]
            imgparts.append(imgp)
    return imgparts,rows,cols

def ImgJoint(imgparts,rows,cols):
    img=np.zeros((rows,cols,3))
    k=0
    for i in range(rows-Height):
        for j in range(cols-Width):
            img[j:j+Width,i:i+Height]+=imgparts[k]
            k=k+1
    maxVal=np.max(img)
    for m in range(rows):
        for n in range(cols):
            #img[n][m] = img[n][m] / ((rows/Height)*(cols/Width)*11225)
            img[n][m] = img[n][m] / (32 * 32 * 225)
    return img


def main():
    # train_data=create_data()
    # k=random.randint(1,1000)
    # net=Network()
    # #net.train(train_data)
    # net.Imgtest(train_data[k])


    img=cv2.imread('images/d.jpg')
    imgparts,rows,cols=ImgDiv(img)
    net = Network()
    outImages = net.test(imgparts)
    print(len(outImages))
    image=ImgJoint(outImages,rows,cols)
    drimage = ImgJoint(imgparts, rows, cols)

    diff=img-image*255

    cv2.imshow("in",img)
    cv2.imshow("drectrecon", drimage)
    cv2.imshow("out",image)
    cv2.imshow("diff", diff)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()