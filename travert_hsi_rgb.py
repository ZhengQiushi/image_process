import cv2 as cv
import numpy as np
import math

'''
    function name: rgb_to_hsi
    parameter:
        img: input image
    return: the img in hsi color domain
'''
def rgb_to_hsi(img):
    row = img.shape[0]
    col = img.shape[1]
    b, g, r = cv.split(img)

    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    ''' generate the hsi domain image '''
    hsi_image = img.copy()

    for i in range(row):
        for j in range(col):
            tmp1 = 1.0 / 2 * ((r[i][j] - g[i][j]) + (r[i][j] - b[i][j]))
            tmp2 = np.sqrt((r[i][j] - g[i][j]) ** 2 + (r[i][j] - b[i][j]) * (g[i][j] - b[i][j]))

            if tmp2 == 0:
                _h = 0
            elif b[i, j] <= g[i, j]:
                _h = np.arccos(tmp1 / tmp2)
            else:
                _h = 2 * np.pi - np.arccos(tmp1 / tmp2)
            _h = _h / (2 * np.pi)

            tmp3 = b[i][j] + g[i][j] + r[i][j]
            if tmp3 == 0:
                _s = 0
            else:
                _s = 1 - 3 * min(min(b[i][j], g[i][j]), r[i][j]) / tmp3

            _i = tmp3 / 3.0

            hsi_image[i][j][0] = _h * 255
            hsi_image[i][j][1] = _s * 255
            hsi_image[i][j][2] = _i * 255
    return hsi_image


'''
    function name: hsi_to_rgb
    parameter:
        img: input image
    return: the img in rgb color domain
'''
def hsi_to_rgb(img):
    row = img.shape[0]
    col = img.shape[1]
    _h, _s, _i = cv.split(img)
    _h = _h / 255.0
    _s = _s / 255.0
    _i = _i / 255.0

    ''' generate the rgb domain image '''
    rgb_img = img.copy()
    for i in range(row):
        for j in range(col):
            if _s[i][j] < 1e-6:
                r = g = b = _i[i][j]
            else:
                _h[i][j] *= 360
                if 0 <= _h[i][j] <= 120:
                    b = _i[i][j] * (1 - _s[i][j])
                    r = _i[i][j] * (1 + (_s[i][j] * math.cos(_h[i][j] * math.pi / 180)) / math.cos(
                        (60 - _h[i][j]) * math.pi / 180))
                    g = 3 * _i[i][j] - (b + r)
                elif 120 < _h[i][j] <= 240:
                    _h[i][j] = _h[i][j] - 120
                    r = _i[i][j] * (1 - _s[i][j])
                    g = _i[i][j] * (1 + (_s[i][j] * math.cos(_h[i][j] * math.pi / 180)) / math.cos(
                        (60 - _h[i][j]) * math.pi / 180))
                    b = 3 * _i[i][j] - (r + g)
                else:
                    _h[i][j] = _h[i][j] - 240
                    g = _i[i][j] * (1 - _s[i][j])
                    b = _i[i][j] * (1 + (_s[i][j] * math.cos(_h[i][j] * math.pi / 180)) / math.cos(
                        (60 - _h[i][j]) * math.pi / 180))
                    r = 3 * _i[i][j] - (g + b)

            rgb_img[i][j][0] = b * 255
            rgb_img[i][j][1] = g * 255
            rgb_img[i][j][2] = r * 255
    return rgb_img


def rgb2hsv(rgb_img):
    """
    :param rgb_img: 输入图像的RGB矩阵
    :return: 输入图像的HSV矩阵
    """
    rgb_img = np.array(rgb_img)
    # 保存原始图像的行列数
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    # 对原始图像进行复制
    hsi_img = rgb_img.copy()
    # 对图像进行归一化并通道拆分
    B, G, R = cv.split(rgb_img / 255)
    # 计算每个像素RGB值的最大最小值
    max_val = np.max(rgb_img, axis=2)
    min_val = np.min(rgb_img, axis=2)
    sub_val = max_val - min_val
    H = np.zeros((row, col))  # 定义H通道
    V = max_val  # 计算V通道
    S = np.where(V == 0, 0, sub_val / V)  # 计算S通道
    H = np.where(V == R, 60 * (G - B) / sub_val, H)
    H = np.where(V == G, 120 + 60 * (B - R) / sub_val, H)
    H = np.where(V == B, 240 + 60 * (R - G) / sub_val, H)
    hsi_img[:, :, 0] = H / 2.0
    hsi_img[:, :, 1] = S * 255.0
    hsi_img[:, :, 2] = V * 255.0
    return hsi_img


def rgb2hsi(rgb_img):
    """
    这是将RGB彩色图像转化为HSI图像的函数
    :param rgb_img: RGB彩色图像
    :return: HSI图像
    """
    # 保存原始图像的行列数
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    # 对原始图像进行复制
    hsi_img = rgb_img.copy()
    # 对图像进行通道拆分
    B, G, R = cv.split(rgb_img)
    # 把通道归一化到[0,1]
    [B, G, R] = [i / 255.0 for i in ([B, G, R])]
    H = np.zeros((row, col))  # 定义H通道
    S = np.zeros((row, col))  # 定义S通道
    I = (R + G + B) / 3.0  # 计算I通道

    # 计算H通道
    for i in range(row):
        den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
        thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)  # 计算夹角
        h = np.zeros(col)  # 定义临时数组
        # den>0且G>=B的元素h赋值为thetha
        h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
        # den>0且G<=B的元素h赋值为thetha
        h[B[i] > G[i]] = 2 * np.pi - thetha[B[i] > G[i]]
        # den=0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h / (2 * np.pi)  # 弧度化后赋值给H通道

    # 计算S通道
    for i in range(row):
        # 找出每组RGB的最小值
        arr = np.c_[B[i], G[i], R[i]]
        min_val = np.min(arr, axis=1)
        # 计算S通道
        S[i] = 1 - min_val * 3 / (R[i] + B[i] + G[i])
        # I为0的值直接赋值0
        S[i][R[i] + B[i] + G[i] == 0] = 0
    # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_img[:, :, 0] = H * 255
    hsi_img[:, :, 1] = S * 255
    hsi_img[:, :, 2] = I * 255
    return hsi_img


def hsi2rgb(hsi_img):
    """
    这是将HSI图像转化为RGB图像的函数
    :param hsi_img: HSI彩色图像
    :return: RGB图像
    """
    # 保存原始图像的行列数
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    # 对原始图像进行复制
    rgb_img = hsi_img.copy()
    # 对图像进行通道拆分
    H, S, I = cv.split(hsi_img)
    # 把通道归一化到[0,1]
    [H, S, I] = [i / 255.0 for i in ([H, S, I])]
    R, G, B = H, S, I
    for i in range(row):
        h = H[i] * 2 * np.pi
        # H大于等于0小于120度时
        a = (0 <= h) == (h < 2 * np.pi / 3)  # 第一种情况的布尔索引
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i] * (1 + S[i] * np.cos(h) / tmp)
        g = 3 * I[i] - r - b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        # H大于等于120度小于240度
        a = (2 * np.pi / 3 <= h) == (h < 4 * np.pi / 3)  # 第二种情况的布尔索引
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i] * (1 + S[i] * np.cos(h - 2 * np.pi / 3) / tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        # H大于等于240度小于360度
        a = (4 * np.pi / 3 <= h) == (h < 2 * np.pi)  # 第三种情况的布尔索引
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1 - S[i])
        b = I[i] * (1 + S[i] * np.cos(h - 4 * np.pi / 3) / tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:, :, 0] = B * 255
    rgb_img[:, :, 1] = G * 255
    rgb_img[:, :, 2] = R * 255
    return rgb_img
