from Image_Process.find_threshold import *
from Image_Process.travert_hsi_rgb import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

'''
    function name: image_enforce
    parameter:
        img: input image ---> gray image [ 0~255 ]
        src_low: the setting low threshold of source image
        src_high: the setting high threshold of source image
        des_low: the setting low threshold of dest image
        des_high: the setting high threshold of dest image
        gama: use gama transformation to proc the enforcement 
    return: the img after enforcing
'''
def image_enforce(img, src_low=-1.0, src_high=-1.0, des_low=-1.0, des_high=-1.0, gama=1.0):
    assert img.ndim == 2
    img_enf = img.copy()
    img_enf = img_enf / 255.0
    if src_low == -1 or src_high == -1:
        src_low, src_high = get_best_threshold(img)
        src_low = src_low / 255.0
        src_high = src_high / 255.0
    if des_low == -1 or des_high == -1:
        des_low = 0.0
        des_high = 1.0

    _gama_high = np.power(src_high, gama)
    _gama_low = np.power(src_low, gama)

    c = (des_high - des_low) / (_gama_high - _gama_low)
    b = des_high - c * _gama_high

    tmp_low = np.argwhere(img_enf < src_low)
    tmp_high = np.argwhere(img_enf > src_high)
    img_enf = np.array((c * np.power(img_enf, gama) + b) * 255, dtype=np.uint8)
    for i in range(len(tmp_low)):
        img_enf[tmp_low[i][0]][tmp_low[i][1]] = img[tmp_low[i][0]][tmp_low[i][1]]

    for i in range(len(tmp_high)):
        img_enf[tmp_high[i][0]][tmp_high[i][1]] = img[tmp_high[i][0]][tmp_high[i][1]]
    return img_enf

'''
    function name: color_image_enforce
    parameter:
        img: input image ---> color image 3 channels
        src_low: the setting low threshold of source image
        src_high: the setting high threshold of source image
        des_low: the setting low threshold of dest image
        des_high: the setting high threshold of dest image
        gama: the coefficient of gama function
    return: the img after enforcing
'''
def color_image_enforce(img, src_low=-1.0, src_high=-1.0, des_low=-1.0, des_high=-1.0, gama=1.0):
    hsi_image = rgb_to_hsi(img)
    h, s, i = cv.split(hsi_image)
    ''' for color image, we should enforcement the intensity segment '''
    i = image_enforce(i, src_low, src_high, des_low, des_high, gama)

    hsi_image = cv.merge(np.array([h, s, i]))
    return hsi_to_rgb(hsi_image)

'''
    function name: color_image_enforce_rgb
    parameter:
        img: input image ---> color image 3 channels
        src_low: the setting low threshold of source image
        src_high: the setting high threshold of source image
        des_low: the setting low threshold of dest image
        des_high: the setting high threshold of dest image
        gama: the coefficient of gama function
    return: the img after enforcing
'''
def color_image_enforce_rgb(img, src_low=-1.0, src_high=-1.0, des_low=-1.0, des_high=-1.0, gama=1.0):
    b, g, r = cv.split(img)
    b = image_enforce(b, src_low, src_high, des_low, des_high, gama)
    g = image_enforce(g, src_low, src_high, des_low, des_high, gama)
    r = image_enforce(r, src_low, src_high, des_low, des_high, gama)
    new_image = cv.merge(np.array([b, g, r]))
    return new_image

def draw(val, name_):
    val = np.array(val)
    val = val.flatten()
    result = np.zeros(256)
    for r in val:
        result[r] += 1
    plt.figure(name_)
    plt.bar(range(256), result)
    plt.show()


"""if __name__ == "__main__":
    img = cv.imread("./2.jpg")
    cv.imshow("1.jpg", img)

    enf = color_image_enforce(img, gama=0.8, des_low=0.10, des_high=0.70)
    draw(enf, "111")
    cv.imshow("2.jpg", enf)

    rgb3 = color_image_enforce_rgb(img, gama=0.8, des_low=0.10, des_high=0.70)
    gray3 = cv.cvtColor(rgb3, cv.COLOR_BGR2GRAY)
    draw(gray3, "222")

    cv.imshow("3.jpg", rgb3)

    cv.waitKey()"""