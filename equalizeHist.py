from find_threshold import *
from travert_hsi_rgb import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def draw(val, name_):
    val = np.array(val)
    val = val.flatten()
    result = np.zeros(256)
    for r in val:
        result[r] += 1
    plt.figure(name_)
    plt.bar(range(256), result)
    plt.show()


def draw_hist(hist, name_):
    '''
        brief@ draw from a hist array
    '''
    plt.figure(name_)
    plt.bar(range(256), hist)
    plt.show()

def getHist(gray_img):
    '''
        brief@ return a hist array from a grayscale image 
    '''
    gray_img = np.array(gray_img)
    gray_img = gray_img.flatten()
    result = np.zeros(256)
    for r in gray_img:
        result[r] += 1
    return result

def calCumHist(src_hist):
    '''
        brief@ calculate the accumulated hist array from the original hist array
    '''
    result = np.zeros(256)
    for r in range(256):
        if r == 0:
            result[r] = src_hist[r]
        else:
            result[r] = result[r - 1] + src_hist[r]
    
    return result



def myEqualHist(gray_img, is_debug = False):
    '''
        brief@ get the image equalized, only for grayscale
        return@ an equalized image!
    '''
    hist_bar = getHist(gray_img)
    if is_debug:
        draw_hist(hist_bar, "ori_hist")

    cum_hist_bar = calCumHist(hist_bar)
    if is_debug:
        draw_hist(cum_hist_bar, "cum_hist")
    
    size = gray_img.shape
    pixel_num = size[0] * size[1]
    pixel_scale = 256

    hist_res = np.zeros(pixel_scale)
    hist_res = (pixel_scale - 1)/pixel_num * cum_hist_bar
    
    res_img = gray_img
    # equalize the pixel
    for x in range(size[0]):
        for y in range(size[1]):
            res_img[x, y] = hist_res[res_img[x, y]]

    if is_debug:
        draw_hist(hist_res, "hist_res")
    
    return res_img


def hsiEqualHist(img):
    '''
        brief@ equalized in hsi style
        return@ 
    '''
    hsi_image = rgb_to_hsi(img)
    h, s, i = cv.split(hsi_image)
    ''' for color image, we should enforcement the intensity segment '''
    i = myEqualHist(i)
    hsi_image = cv.merge(np.array([h, s, i]))
    return hsi_to_rgb(hsi_image)

def rgbEqualHist(img):
    '''
        brief@ equalized in rgb style
        return@ 
    '''
    b, g, r = cv.split(img)
    b = myEqualHist(b)
    g = myEqualHist(g)
    r = myEqualHist(r)
    new_image = cv.merge(np.array([b, g, r]))
    return new_image


if __name__ == "__main__":
    img = cv.imread("./2.jpg")
    cv.imshow("1.jpg", img)

    rgb3 = color_image_enforce_rgb(img, gama=0.8, des_low=0.10, des_high=0.70)
    gray3 = cv.cvtColor(rgb3, cv.COLOR_BGR2GRAY)
    draw(gray3, "222")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    enf = cv.equalizeHist(gray)
    my_res = myEqualHist(gray)


    my_rgb_res = rgbEqualHist(img)
    my_hsi_res = hsiEqualHist(img)



    cv.imshow("2.jpg", enf)
    draw(enf, "111")

    cv.imshow("my", my_res)
    draw(my_res, "121")

    cv.imshow("my_rgb", my_rgb_res)
    draw(my_res, "131")

    cv.imshow("my_hsi", my_hsi_res)
    draw(my_hsi_res, "141")

    # rgb3 = color_image_enforce_rgb(img, gama=0.8, des_low=0.10, des_high=0.70)
    # gray3 = cv.cvtColor(rgb3, cv.COLOR_BGR2GRAY)
    # draw(gray3, "222")

    # cv.imshow("3.jpg", rgb3)

    cv.waitKey()