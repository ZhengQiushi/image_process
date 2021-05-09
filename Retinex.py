#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/5/7 00:54
# @Author : Jerry Liu

import numpy as np
import random
import cv2 as cv


def replaceZeroes(data):
    """
    :param data: 输入的数据
    :return: 去除0之后的数据
    """
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def salt_and_pepper_noise(img, proportion=0.05):
    """
    :param img: 原图像矩阵
    :param proportion: 噪点的占比
    :return: 添加噪点后的矩阵
    """
    noise_img = img
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def median_filter(data, win=3):
    """
    :param data: 输入的数据，二维np.array
    :param win: 卷积核的大小，奇数
    :return: 中值滤波后的矩阵
    """
    # data = np.array(data)
    H, W = data.shape
    result = data.copy()
    add_val = win // 2
    data = np.pad(data, ((add_val, add_val), (add_val, add_val)), 'edge')
    for h in range(0, H):
        for w in range(0, W):
            result[h, w] = np.median(data[h:h + win, w:w + win])
    return result


def SSR(src_img, win=3):
    """
    使用中值滤波
    :param data: 输入的数据，二维np.array
    :param win: 卷积核的大小，奇数
    :return: SSR算法之后的矩阵
    """
    L_blur = median_filter(src_img, win)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv.log(img / 255.0)
    dst_Lblur = cv.log(L_blur / 255.0)
    dst_IxL = cv.multiply(dst_Img, dst_Lblur)
    log_R = cv.subtract(dst_Img, dst_IxL)

    dst_R = cv.normalize(log_R, None, 0, 255, cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8


def SSR2(src_img, size):
    """
    使用高斯滤波
    :param src_img: 输入的图像
    :param size: 卷积核大小
    :return: SSR算法之后的矩阵
    """
    L_blur = cv.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)
    dst_Img = cv.log(img / 255.0)
    dst_Lblur = cv.log(L_blur / 255.0)
    dst_IxL = cv.multiply(dst_Img, dst_Lblur)
    log_R = cv.subtract(dst_Img, dst_IxL)
    dst_R = cv.normalize(log_R, None, 0, 255, cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8