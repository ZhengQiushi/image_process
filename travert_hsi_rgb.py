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
                    r = _i[i][j] * (1 + (_s[i][j] * math.cos(_h[i][j] * math.pi / 180)) / math.cos((60 - _h[i][j]) * math.pi / 180))
                    g = 3 * _i[i][j] - (b + r)
                elif 120 < _h[i][j] <= 240:
                    _h[i][j] = _h[i][j] - 120
                    r = _i[i][j] * (1 - _s[i][j])
                    g = _i[i][j] * (1 + (_s[i][j] * math.cos(_h[i][j] * math.pi / 180)) / math.cos((60 - _h[i][j]) * math.pi / 180))
                    b = 3 * _i[i][j] - (r + g)
                else:
                    _h[i][j] = _h[i][j] - 240
                    g = _i[i][j] * (1 - _s[i][j])
                    b = _i[i][j] * (1 + (_s[i][j] * math.cos(_h[i][j] * math.pi / 180)) / math.cos((60 - _h[i][j]) * math.pi / 180))
                    r = 3 * _i[i][j] - (g + b)

            rgb_img[i][j][0] = b * 255
            rgb_img[i][j][1] = g * 255
            rgb_img[i][j][2] = r * 255
    return rgb_img

if __name__ == "__main__":
    img = cv.imread("./1.jpg")
    cv.imshow("1.jpg", img)
    hsi = rgb_to_hsi(img)
    cv.imshow("2.jpg", hsi)
    cv.imshow("3.jpg", hsi_to_rgb(hsi))
    cv.waitKey()