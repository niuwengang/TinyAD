import cv2
import numpy as np
# 打印OpenCV版本
print(cv2.getVersionString())


#颜色通道
def func1():
    image=cv2.imread("data/picture/opencv_logo.jpg")
    cv2.imshow("blue",image[:,:,0])
    cv2.imshow("green",image[:,:,1])
    cv2.imshow("red",image[:,:,2])
    cv2.waitKey()

#灰度化
def func2():
    image=cv2.imread("data/picture/opencv_logo.jpg")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    cv2.waitKey()

#裁减
def func3():
    image=cv2.imread("data/picture/opencv_logo.jpg")
    crop=image[10:40,40:80]
    cv2.imshow("crop",crop)
    cv2.waitKey()

#绘制图样
def func4():
    image=np.zeros([300,300,3],dtype=np.uint8)
    cv2.line(image,(100,200),(250,250),(255,0,0),2)
    cv2.rectangle(image,(30,100),(60,150),(0,255,0),2)
    cv2.circle(image,(150,100),20,(0,0,255),2)
    cv2.putText(image,"hello",(100,50),0,1,(255,255,255),2,1)
    cv2.imshow("image",image)
    cv2.waitKey()

#图像滤波
def func5():
    image=cv2.imread("data/picture/plane.jpg")
    gauss=cv2.GaussianBlur(image,(5,5),0)
    median=cv2.GaussianBlur(image,(5,5),0)
    cv2.imshow("image",image)
    cv2.imshow("gauss",gauss)
    cv2.imshow("median",median)
    cv2.waitKey()

#特征提取
def func6():
    image=cv2.imread("data/picture/opencv_logo.jpg")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gray,500,0.1,10)
    for corner in corners:
        x,y=corner.ravel()
        cv2.circle(image,(int(x),int(y)),3,(255,0,255),-1)
    cv2.imshow("image",image)
    cv2.waitKey()



#腐蚀与膨胀
def func7():
    image=cv2.imread("data/picture/opencv_logo.jpg")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    kernel=np.ones((5,5),np.uint8)
    erosion=cv2.erode(binary,kernel)
    dilation=cv2.dilate(binary,kernel)
    cv2.imshow("binary",binary)
    cv2.imshow("erosion",erosion)
    cv2.imshow("dilation",dilation)
    cv2.waitKey()

#摄像头调用
def func8():
    capture=cv2.VideoCapture(0)
    while True:
        ret,frame=capture.read()
        cv2.imshow("camera",frame)
        key=cv2.waitKey(1)
        if key !=-1:
            break
    capture.release()

#模板匹配
def func9():
    image=cv2.imread("data/picture/poker.jpg")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    template=gray[75:105,235:265]
    match=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
    locs=np.where(match>=0.9)
    w,h=template.shape[0:2]
    for p in zip(*locs[::-1]):
        x1,y1=p[0],p[1]
        x2,y2=x1+w,y1+h
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),2)
    cv2.imshow("image",image)
    cv2.waitKey()


if __name__ == "__main__":
    # func1()
    # func2()
    # func3()
    # func4()
    # func5()
    func9()
