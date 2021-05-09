import numpy as np
import cv2 as cv
import math
from find_threshold import *
from travert_hsi_rgb import *
from equalizeHist import *

import matplotlib.pyplot as plt

'''
    notice : 1. tile_size should be square like [32, 32]! rect maybe incorrect!!
             2. there would be some margins which remains the same! Try to resize or clip the image into the muliply of the tile_size will solve the problem...
             Orz
'''


def clipHist(src_hist, src_img, clip_limit, grayscale=256):
    '''
        brief@ CLHE clip the pixel which is more than the threshold
                    then spread the clipped num evenly
        params@ src_hist(a 256 hist_array)
                src_img(a grayscale image)
                clip_limit(a coeffient...)
        return@ a 256 hist_array which has been clipped
    '''
    img_size = src_img.shape
    img_width = img_size[0]
    img_height = img_size[1]
    # calculate the threshold 
    # average = img_width * img_height / grayscale
    # clip_thresh = clip_limit * average
    clip_thresh = clip_limit

    clipped_pixel_num = 0
    for i in range(grayscale):
        # clip the one which is higher than the threshold
        if src_hist[i] > clip_thresh:
            clipped_pixel_num += src_hist[i] - clip_thresh
            src_hist[i] = clip_thresh
    # spread it evenly
    bonus_aver = clipped_pixel_num / grayscale
    for i in range(grayscale):
        src_hist[i] += bonus_aver

    return src_hist

def clhe(gray_img, is_debug = False):
    '''
        brief@ get the image equalized, only for grayscale
        return@ an equalized image!
    '''
    hist_bar = getHist(gray_img)
    if is_debug:
        draw_hist(hist_bar, "ori_hist")

    hist_bar = clipHist(hist_bar, gray_img, 200)
    if is_debug:
        draw_hist(hist_bar, "cliped_hist")

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


def claheWithoutInterpolation(gray_img, tile_size, clip_limit, pixel_scale=256):
    '''
        brief@  WithoutInterpolation
                1. first cut the image into several clipping window according to the tile_size
                2. cal hist_array and accumulated hist_array for each of the clipping window
                3. remapping the clipping window to the original image
        params@ gray_img( a grayscale image )
                tile_size( size of the clipping window i.e. [32,32] )
                clip_limit( a coeffient... )
        return@ result image
    '''
    src = gray_img
    img_size = gray_img.shape
    # attribute for certain block
    block_width = tile_size[0]
    block_height = tile_size[1]

    block_width_num = img_size[0] // block_width
    block_height_num = img_size[1] // block_height
    block_pixel_num = block_width * block_height

    total_block = block_width_num * block_height_num
    # hist_block[2, : ] stands for the hist_array for the second clipping window!
    hist_block = np.zeros((total_block, pixel_scale))
    cum_block = np.zeros((total_block, pixel_scale))
    final_hist_block = np.zeros((total_block, pixel_scale))

    # cal hist and accumulated for each block
    for i in range(block_width_num):
        for j in range(block_height_num):
            # location and span
            start_x = i * block_width
            start_y = j * block_height
            end_x = start_x + block_width
            end_y = start_y + block_height
            cur_num = i * block_height_num + j

            # clip the image
            sub_block_img = gray_img[start_x:end_x, start_y: end_y]
            sub_hist = getHist(sub_block_img)
            # clip the pixel
            sub_clipped_hist = clipHist(sub_hist, sub_block_img, clip_limit)
            # cal the accumulated hist_array
            sub_cum_hist = calCumHist(sub_clipped_hist)

            hist_block[cur_num, :] = sub_hist
            cum_block[cur_num, :] = sub_cum_hist
            final_hist_block[cur_num, :] = (pixel_scale - 1) / block_pixel_num * sub_cum_hist

    # remapping !
    for i in range(block_width_num):
        for j in range(block_height_num):
            # location and span
            start_x = i * block_width
            start_y = j * block_height
            end_x = start_x + block_width
            end_y = start_y + block_height

            cur_num = i * block_height_num + j
            # remapping the image
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    src[x, y] = final_hist_block[cur_num, src[x, y]]

    return src


def claheWithInterpolation(gray_img, tile_size=None, clip_limit=4, pixel_scale=256):
    '''
        brief@  WithoutInterpolation
                1. first cut the image into several clipping windows according to the tile_size
                2. cal hist_array and accumulated hist_array for each of the clipping window
                3. do interpolation for all clipping windows
                4. remapping the clipping window to the original image
        params@ gray_img( a grayscale image )
                tile_size( size of the clipping window i.e. [32,32] )
                clip_limit( a coeffient... )
        return@ result image
    '''
    if tile_size is None:
        tile_size = [32, 32]
    src = gray_img
    img_size = gray_img.shape
    img_width = img_size[0]
    img_height = img_size[1]
    # attribute for certain block
    block_width = tile_size[0]
    block_height = tile_size[1]

    img_width = img_width // block_width * block_width
    img_height = img_height // block_height * block_height

    block_width_num = img_size[0] // block_width
    block_height_num = img_size[1] // block_height
    block_pixel_num = block_width * block_height

    total_block = block_width_num * block_height_num

    hist_block = np.zeros((total_block, pixel_scale))
    cum_block = np.zeros((total_block, pixel_scale))
    final_hist_block = np.zeros((total_block, pixel_scale))

    # cal hist and accumulated for each block
    for i in range(block_width_num):
        for j in range(block_height_num):
            # location and span
            start_x = i * block_width
            start_y = j * block_height
            end_x = start_x + block_width
            end_y = start_y + block_height

            cur_num = i * block_height_num + j

            # clip the image
            sub_block_img = gray_img[start_x:end_x, start_y: end_y]
            sub_hist = getHist(sub_block_img)
            # clip the pixel
            sub_clipped_hist = clipHist(sub_hist, sub_block_img, clip_limit)

            sub_cum_hist = calCumHist(sub_clipped_hist)

            hist_block[cur_num, :] = sub_hist
            cum_block[cur_num, :] = sub_cum_hist
            final_hist_block[cur_num, :] = (pixel_scale - 1) / block_pixel_num * sub_cum_hist

    sub_block_width = block_width // 2
    sub_block_height = block_height // 2

    for m in range(img_height):
        for n in range(img_width):
            # four coners : directly remapping
            if m <= block_height / 2 and n <= block_width / 2:
                block_m = 0
                block_n = 0
                src[n, m] = final_hist_block[block_m + block_n * block_height_num, src[n, m]]
            elif m <= block_height / 2 and n >= img_width - block_width / 2:
                block_m = 0
                block_n = block_width_num - 1
                src[n, m] = final_hist_block[block_m + block_n * block_height_num, src[n, m]]
            elif n <= block_width / 2 and m >= img_height - block_height / 2:
                block_m = block_height_num - 1
                block_n = 0
                src[n, m] = final_hist_block[block_m + block_n * block_height_num, src[n, m]]
            elif m >= img_height - block_height / 2 and n >= img_width - block_width / 2:
                block_m = block_height_num - 1
                block_n = block_width_num - 1
                src[n, m] = final_hist_block[block_m + block_n * block_height_num, src[n, m]]
            # four edges except coners : linear interpolation
            elif n <= block_width / 2:
                block_m_1 = math.floor((m - block_height / 2) / block_height)
                block_n_1 = 0
                block_m_2 = block_m_1 + 1
                block_n_2 = block_n_1
                u = np.float64((m - (block_m_1 * block_height + block_height / 2)) / (block_height))
                v = 1 - u
                src[n, m] = v * final_hist_block[block_m_1 + block_n_1 * block_height_num, src[n, m]] + \
                            u * final_hist_block[block_m_2 + block_n_2 * block_height_num, src[n, m]]
            elif m <= block_height / 2:
                block_m_1 = 0
                block_n_1 = math.floor((n - block_width / 2) / block_width)
                block_m_2 = block_m_1
                block_n_2 = block_n_1 + 1
                u = np.float64((n - (block_n_1 * block_width + block_width / 2)) / (block_width))
                v = 1 - u
                src[n, m] = v * final_hist_block[block_m_1 + block_n_1 * block_height_num, src[n, m]] + \
                            u * final_hist_block[block_m_2 + block_n_2 * block_height_num, src[n, m]]

            elif m >= img_height - block_height / 2:
                block_m_1 = block_height_num - 1
                block_n_1 = math.floor((n - block_width / 2) / block_width)
                block_m_2 = block_m_1
                block_n_2 = block_n_1 + 1
                u = np.float64((n - (block_n_1 * block_width + block_width / 2)) / (block_width))
                v = 1 - u
                src[n, m] = v * final_hist_block[block_m_1 + block_n_1 * block_height_num, src[n, m]] + \
                            u * final_hist_block[block_m_2 + block_n_2 * block_height_num, src[n, m]]
            elif n >= img_width - block_width / 2:
                block_m_1 = math.floor((m - block_height / 2) / block_height)
                block_n_1 = block_width_num - 1
                block_m_2 = block_m_1 + 1
                block_n_2 = block_n_1
                u = np.float64((m - (block_m_1 * block_height + block_height / 2)) / (block_height))
                v = 1 - u
                src[n, m] = v * final_hist_block[block_m_1 + block_n_1 * block_height_num, src[n, m]] + \
                            u * final_hist_block[block_m_2 + block_n_2 * block_height_num, src[n, m]]
            else:
                # content : double linear interpolation
                block_m_1 = math.floor((m - block_height / 2) / block_height)
                block_n_1 = math.floor((n - block_width / 2) / block_width)
                block_m_2 = block_m_1 + 1
                block_n_2 = block_n_1 + 1
                u = np.float64((m - (block_m_1 * block_height + block_height / 2)) / (block_height))
                v = np.float64((n - (block_n_1 * block_width + block_width / 2)) / (block_width))

                src[n, m] = (1 - u) * (1 - v) * final_hist_block[block_m_1 + block_n_1 * block_height_num, src[n, m]] + \
                            v * (1 - u) * final_hist_block[block_m_1 + block_n_2 * block_height_num, src[n, m]] + \
                            u * (1 - v) * final_hist_block[block_m_2 + block_n_1 * block_height_num, src[n, m]] + \
                            u * v * final_hist_block[block_m_2 + block_n_2 * block_height_num, src[n, m]]

    return src


def hsiClahe(img):
    '''
        brief@ equalized in hsi style
        return@ 
    '''
    hsi_image = rgb_to_hsi(img)
    h, s, i = cv.split(hsi_image)
    ''' for color image, we should enforcement the intensity segment '''
    i = claheWithInterpolation(i)
    hsi_image = cv.merge(np.array([h, s, i]))
    return hsi_to_rgb(hsi_image)


def rgbClahe(img):
    '''
        brief@ equalized in rgb style
        return@ 
    '''
    b, g, r = cv.split(img)
    b = claheWithInterpolation(b)
    g = claheWithInterpolation(g)
    r = claheWithInterpolation(r)
    new_image = cv.merge(np.array([b, g, r]))
    return new_image


if __name__ == "__main__":
    img = cv.imread("./1.jpg")
    
    gray_img0 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # my_clhe = clhe(gray_img0, True)

    myEqualHist(gray_img0, True)

    # cv.imshow("gray_img0", my_clhe)
    # draw(my_clhe, "gray_img0")


    gray_img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    my_clahe_res_without_interpolation = claheWithoutInterpolation(gray_img1, [32, 32], 4)
    cv.imshow("my_clahe_res_without_interpolation", my_clahe_res_without_interpolation)
    draw(my_clahe_res_without_interpolation, "my_clahe_res_without_interpolation_hist")

    gray_img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    my_clahe_res = claheWithInterpolation(gray_img2, [32, 32], 4)
    cv.imshow("my_clahe_res", my_clahe_res)
    draw(my_clahe_res, "my_clahe_res")

    gray_img3 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))
    cv_clahe_res = clahe.apply(gray_img3)
    cv.imshow("cv_clahe_res", cv_clahe_res)
    draw(cv_clahe_res, "cv_clahe_res_hist")

    my_rgb_res = rgbClahe(img)
    my_hsi_res = hsiClahe(img)

    cv.imshow("my_rgb", my_rgb_res)
    cv.imshow("my_hsi", my_hsi_res)



    cv.waitKey()
