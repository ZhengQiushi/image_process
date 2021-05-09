#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/5/7 00:54
# @Author : Niannian Zhong

import numpy as np
import cv2

"""
    非局部均值降噪算法的实现
"""


def arraycompare(array1, array2, height, width):
    """
    :param array1: 输入的二维list1
    :param array2: 输入的二维list2
    :param height: 两个list的高
    :param width: 两个list的宽
    :return: 返回每个位置上的较大值
    """
    resultarray = np.zeros((height, width))
    for row in range(0, height):
        for col in range(0, width):
            resultarray[row, col] = max(array1[row, col], array2[row, col])
    return resultarray


def integralImgSqDiff2(paddedimg_val, Ds_val, t1_val, t2_val):
    """
    :param paddedimg_val: 搜索矩阵x
    :param Ds_val: 搜索窗口大小
    :param t1_val: t1的值
    :param t2_val: t2的值
    :return: 去噪后像素点的灰度值
    """
    lengthrow = len(paddedimg_val[:, 0])
    lengthcol = len(paddedimg_val[0, :])
    Dist2 = (paddedimg_val[Ds_val:-Ds_val, Ds_val:-Ds_val] -
             paddedimg_val[Ds_val + t1_val:lengthrow - Ds_val + t1_val,
                           Ds_val + t2_val:lengthcol - Ds_val + t2_val]) ** 2
    Sd_val = Dist2.cumsum(0)
    Sd_val = Sd_val.cumsum(1)
    return Sd_val


def nl_meansfilter(imagearray, h_=10, ds0=2, ds1=5):
    """
    :param imagearray: 输入图像的RGB矩阵
    :return: 去噪后图像的RGB矩阵
    """
    # print("非局部均值降噪算法开始")
    height, width = imagearray[:, :, 0].shape[:2]  # 获取图像的宽和高
    length0 = height + 2 * ds1  # 对边缘进行扩展之后的高
    length1 = width + 2 * ds1  # 对边缘进行扩展之后的宽
    h = (h_ ** 2)
    d = (2 * ds0 + 1) ** 2
    imagearray_NL = np.zeros(imagearray.shape).astype('uint8')
    for i in range(0, 3):
        # print(i)
        paddedimg = np.pad(imagearray[:, :, i], ds0 + ds1 + 1, 'symmetric')
        paddedimg = paddedimg.astype('float64')
        paddedv = np.pad(imagearray[:, :, i], ds1, 'symmetric')
        paddedv = paddedv.astype('float64')
        average = np.zeros((height, width))
        sweight = np.zeros((height, width))
        wmax = np.zeros((height, width))
        for t1 in range(-ds1, ds1 + 1):
            for t2 in range(-ds1, ds1 + 1):
                if t1 == 0 and t2 == 0:
                    continue
                # print(t1, t2)
                Sd = integralImgSqDiff2(paddedimg, ds1, t1, t2)
                SqDist2 = Sd[2 * ds0 + 1:-1, 2 * ds0 + 1:-1] + Sd[0:-2 * ds0 - 2, 0:-2 * ds0 - 2] - \
                          Sd[2 * ds0 + 1:-1, 0:-2 * ds0 - 2] - Sd[0:-2 * ds0 - 2, 2 * ds0 + 1:-1]
                SqDist2 /= d * h
                w = np.exp(-SqDist2)
                v = paddedv[ds1 + t1:length0 - ds1 + t1, ds1 + t2:length1 - ds1 + t2]
                average += w * v
                wmax = arraycompare(wmax, w, height, width)
                sweight += w
        average += wmax * imagearray[:, :, i]
        average /= wmax + sweight
        average_uint8 = average.astype('uint8')
        imagearray_NL[:, :, i] = average_uint8
    return imagearray_NL


"""
test = cv2.imread("1.png", cv2.IMREAD_COLOR)
NL_ = nl_meansfilter(test)
cv2.imshow("true", test)
cv2.imshow("NL_", NL_)
cv2.waitKey()
"""
