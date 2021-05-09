

# 说明文档

###  0.文档背景

本仓库代码从底层完成了论文《基于色彩空间分解的低照度图像增强方法》的复现。

![image-20210509220933348](C:\Users\THINK\AppData\Roaming\Typora\typora-user-images\image-20210509220933348.png)

### 1.文件结构

- dataset:数据集
- result:处理结果

- clahe.py 
  - def clhe ： 实现CLHE算法（饱和度限制直方图均衡化）
  - def claheWithoutInterpolation ： 实现不带插值的CLAHE算法
  - def claheWithInterpolation: 实现带插值的CLAHE算法
- equalizeHist.py
  - def myEqualHist： 实现直方图均衡化算法
- find_threshold.py
  - def get_best_threshold：实现自适应寻找图像映射阈值算法
- image_enforce.py
  - def image_enforce：实现直方图灰度图图像增强算法
  - def color_image_enforce：实现直方图彩色图像增强算法
- midterm_homework.py
  - def run_main：主函数👈
- NL_means.py
  - def nl_meansfilter：实现非局部均值降噪算法
- Retinex.py
  - def SSR：中值滤波
  - def SSR2：高斯滤波
- travert_hsi_rgb.py
  - def rgb_to_hsi：将图像从RGB颜色空间转换到HSI颜色空间
  - def hsi_to_rgb：将图像从HSI颜色空间转换到RGB颜色空间

## 2.运行环境

python 3.7 + windows10

## 3. API 与调用说明

#### 直接主程序运行：

```
> python3 midterm_homework.py
```

即可以得到最终运行结果（左1原图 中间rgb颜色空间的增强效果 右1hsi颜色空间的增强效果）：

（由于是自己实现的程序，所以运行效率不是很高，请耐心等待...）

<img src="C:\Users\THINK\AppData\Roaming\Typora\typora-user-images\image-20210509223114307.png" alt="image-20210509223114307" style="zoom:33%;" /><img src="C:\Users\THINK\AppData\Roaming\Typora\typora-user-images\image-20210509223123615.png" alt="image-20210509223123615" style="zoom:33%;" /><img src="C:\Users\THINK\AppData\Roaming\Typora\typora-user-images\image-20210509223134397.png" alt="image-20210509223134397" style="zoom:33%;" />

我们的数据集存放在`./dataset`中，运行结果存放再`./result`中

```python
  data_path = "./dataset" # 数据来源
  res_path = "./result" # 运行结果

  filelist = [i for i in os.listdir(data_path)]
  for file in filelist:
      res_img = run_main(data_path + "/" + file)
      cv.imwrite(res_path + "/" + file, res_img)
      print("finish :" + file)
      cv.waitKey()
```

默认的缩放大小是`100*100`，颜色空间是`rgb`，不打印每一步的图片输出结果（`is_debug`）

```
def run_main(img_dir,img_size =(100,100),is_rgb = True, is_debug = False):
```

#### 您也可以单独运行各个部分;

**非局部均值降噪算法**

```python
test = cv2.imread("1.png", cv2.IMREAD_COLOR)
NL_ = nl_meansfilter(test) 
cv2.imshow("原图", test)
cv2.imshow("降噪后", NL_)
```

**直方图灰度/彩色图图像增强算法**

```python
# 灰度增强
img = cv.imread("./2.jpg")
enf = color_image_enforce(img, gama=0.8, des_low=0.10, des_high=0.70)
draw(enf, "灰度增强后直方图")
cv.imshow("2.jpg", enf)

# 彩色增强
rgb3 = color_image_enforce_rgb(img, gama=0.8, des_low=0.10, des_high=0.70)
gray3 = cv.cvtColor(rgb3, cv.COLOR_BGR2GRAY)
draw(gray3, "彩色增强后直方图")
cv.imshow("彩色增强", rgb3)
```

**带插值的CLAHE算法**

```python
# RGB颜色空间内的CLAHE算法
my_rgb_res = rgbClahe(img)
cv.imshow("my_rgb", my_rgb_res)

# HSI颜色空间内的CLAHE算法
my_hsi_res = hsiClahe(img)
cv.imshow("my_hsi", my_hsi_res)

# 灰度空间内的CLAHE算法
gray_img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
my_clahe_res = claheWithInterpolation(gray_img2, [32, 32], 4)
cv.imshow("my_clahe_res", my_clahe_res)
# 画出某灰色图的直方图
draw(my_clahe_res, "my_clahe_res")
```

**直方图均衡化算法**

```python
# RGB颜色空间内的HE算法
my_rgb_res = rgbEqualHist(img)
# HSI颜色空间内的HE算法
my_hsi_res = hsiEqualHist(img)
```

**画出直方图**

```python
# 画出某灰色图的直方图
draw(my_clahe_res, "my_clahe_res")
```

