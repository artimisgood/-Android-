# -Android-
# 对于计算机视觉的理解——Android实验室第一次培训
大一寒假Android实验室培训了这方面的内容。第一次写技术类的贴，这是一篇记录的贴（也是实验室作业，这种被推着奋进的感觉不错但是还得培养自己的主观能动性），不太熟练，总之咱们先开始吧。




#### 计算机视觉是什么？
视觉——即与图像处理有关->图像处理相对于其他数据的处理更为复杂->机器学习->上升到三维空间里。



#### 计算机视觉与计算机图形学的区别在哪？
其实所用处理方法相通，但是计算机图形学更注重美感。




## openCV初步学习

- 来段代码：
```c++
#include<iostream>//必要前缀
#include<opencv2/opencv.hpp>//必要前缀
#define Pi 3.1415926

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{

  VideoCapture video(0);//摄像头捕捉信息(这个做起来还是相当有意思的)
  while (1)
  {
    Mat frame;
    video >> frame;
    cvtColor(frame, frame, COLOR_RGB2GRAY);
    namedWindow("frame",CV_WINDOW_AUTOSIZE);
    imshow("frame", frame);
    waitKey(30);//每三十毫秒刷新一次
  }

	Mat src = imread("F:/girl.jpg", 1);//注意反斜杠
	cvtColor(src, src, COLOR_BGR2GRAY);//将图转变成黑白
	namedWindow("src", WINDOW_AUTOSIZE);//新建窗口
	imshow("src", src);
	waitKey(0);
  return 0;
}
```

- 这里还有一些opencv与c的一些不同用法：
>char-UCHAR          int-CV32S           float-CV32S          double-CV64F

>
现在opencv的最基础用法我们已经熟悉的差不多了。


## 做一个例子：实现图像X方向一阶差分，观察图像的边缘提取情况

```c++
#include<iostream>//必要前缀
#include<opencv2/opencv.hpp>//必要前缀
#define Pi 3.1415926

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
  Mat dImg = Mat(src.rows, src.cols - 2, CV_8UC1);//C1 = CHANNAL,差分！！
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			dImg.at<uchar>(i, j - 1) = src.at<uchar>(i, j + 1) - src.at<uchar>(i, j - 1);
		}
	}//图像只有单方向，卷积核也只有单方向，所以需要两层循环来进行遍历

	namedWindow("dst", CV_WINDOW_AUTOSIZE);
	imshow("dst",dImg);
	waitKey(0);
	Mat src = imread("D:/a.jpg", 1);//注意反斜杠
	cvtColor(src, src, COLOR_BGR2GRAY);//将图转变成黑白
	namedWindow("src",WINDOW_AUTOSIZE);//新建窗口
	imshow("src", src);
  return 0;
}
```

---
## 我们明确几点概念：
#### 1.图像
计算机产生的图像实际上是由三个矩阵产生的。也可以看作一组波。



#### 2.取样和量化
我们可以将图像块状分割，从每块取样。采样点越少，图片越模糊。                                        
![image](https://github.com/artimisgood/-Android-/blob/master/1.jpg)



#### 3.灰度级
离散化后的像素值用一个字节表示（8位）。颜色模式RGB每个字母所代表的程度都用一个字节表示。也就是说一共有256*256*256种颜色。
![image](https://github.com/artimisgood/-Android-/blob/master/2.jpg)


---
### 既然图像还可以是一组波，那么我们将图片离散化的时候是否也可以将波联系起来呢？                           
我们提出傅里叶级数的概念。   
![image](https://github.com/artimisgood/-Android-/blob/master/图片3.png)
其本质就是把周期信号波转化为无限多个离散的正弦波。                      
![image](https://github.com/artimisgood/-Android-/blob/master/图片4.png)




---
### 我们提出一个问题：如果一张图里有白雪公主，我们如何把白雪公主拿出来并且不破坏背景图？
我们应了解傅里叶变化
![image](https://github.com/artimisgood/-Android-/blob/master/图片5.png)
![image](https://github.com/artimisgood/-Android-/blob/master/图片6.png)
- 如图，进行傅里叶级数计算后，我们从空间域视角转换为频率域视角。其实，白雪公主可能就是其中的一条，我们只要从频率域视角中选出来并且剔除掉，就可能实现问题中的效果。 

---
### 接下来我们又提出了一个问题。如何去除下图中的噪点？？                
![image](https://github.com/artimisgood/-Android-/blob/master/图片7.png)
- 我们又认识到了一个新概念，图像滤波。图像滤波分为两种：平滑化滤波和锐化滤波。                                 
 * 以上铅笔图中，有很多杂点，我们称之为噪点。平滑化滤波可以去噪，也就是去除噪点。噪点的像素值与周围的像素值非常不同，而去噪就是让其像素值变得与周围相同。锐化恰恰相反，他能让图像的边界变得更加明显。
#### 接下来我们必须熟悉一下图像求导和图像微分的知识，才能更好的编写代码
我们先类比连续函数求导：                                
![image](https://github.com/artimisgood/-Android-/blob/master/图片7.png)
这是图像求导：我们可以使用有限差分表示图像的导数或者偏导数（离散数学没学到所以不是很理解）。
![image](https://github.com/artimisgood/-Android-/blob/master/图片8.png)
这里使用差分近似的。

