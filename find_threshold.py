'''
    find the best threshold of an img
'''

import cv2 as cv
import numpy as np

'''
    function name: get_img_GreyLayer
    parameter:
        img: input image
    return: the ndarray of img's GreyLayer
'''
def get_img_GreyLayer(img):
    res = np.zeros(256, dtype=np.uint64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[img[i][j]] += 1
    return res

'''
    function name: get_img_pdf
    parameter:
        img: input image
        grey_layer: the ndarray of img's GreyLayer
    return: the ndarray of img's pdf probability density function
'''
def get_img_pdf(img, grey_layer):
    pdf = np.zeros(256, dtype=np.float64)
    cnt = img.shape[0] * img.shape[1]
    pdf = grey_layer / cnt
    return pdf

'''
    function name: get_img_cdf
    parameter:
        pdf: the ndarray of img's pdf
    return: the ndarray of img's cdf cumulative distribution function
'''
def get_img_cdf(pdf):
    cdf = np.zeros(256, dtype=np.float64)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    return cdf

'''
    function name: get_best_threshold
    parameter:
        img: input image
        set_low: the setting lower threshold
        set_high: the setting higher threshold
    return: the list of best threshold 
'''
def get_best_threshold(img, set_low=0.05, set_high=0.95):
    assert img.ndim == 2
    ''' compute the GreyLayer of img '''
    grey_layer = get_img_GreyLayer(img)
    ''' compute the pdf of img'''
    pdf = get_img_pdf(img, grey_layer)
    ''' compute the cdf of img '''
    cdf = get_img_cdf(pdf)

    thr_low = np.where(cdf >= set_low)[0][0]
    thr_high = np.where(cdf >= set_high)[0][0]

    if thr_low == thr_high:
        print("最佳low值与最佳high值相同，返回全部")
        return 0, 1
    else:
        return thr_low, thr_high


"""if __name__ == "__main__":
    img = cv.imread("C:/Users/asus/Desktop/1.jpg")
    cv.imshow("1.jpg", img.copy())
    cv.waitKey()"""

