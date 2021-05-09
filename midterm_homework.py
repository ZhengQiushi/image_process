#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/5/5 14:33
# @Author : Jerry Liu

import cv2 as cv
from Image_Process.Retinex import SSR, SSR2
from Image_Process.clahe import claheWithInterpolation
from Image_Process.NL_means import nl_meansfilter
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt


def run_main():
    """
    这是主函数
    """
    # 利用opencv读入图片
    rgb_img = cv.imread('2.jpg')
    cv.imshow("truth", rgb_img)
    # 进行颜色空间转换
    hsv_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv_img)
    V = SSR(V, 5)

    ret1, th1 = cv.threshold(S, 0, 255, cv.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    S = claheWithInterpolation(S, clip_limit=ret1 * 0.7)

    hsv_img[:, :, 0] = H
    hsv_img[:, :, 1] = S
    hsv_img[:, :, 2] = V

    hsv_img = nl_meansfilter(hsv_img)
    rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
    cv.imshow("result", rgb_img)
    cv.imshow("result1", hsv_img)

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
  run_main()



