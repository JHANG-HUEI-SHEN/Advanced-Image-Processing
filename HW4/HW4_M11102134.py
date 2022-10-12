#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget,QPushButton,QFileDialog,QGridLayout,QLabel,QInputDialog,QLineEdit,QTextEdit)
from PyQt5.QtGui import QImage, QPixmap

def conv(img, kernel,model):
        img=np.pad(img,((2,2),(2,2)),'edge')#鏡像邊緣   
        size = img.shape[0]-kernel.shape[0]+1
        new_img = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                output = 0
                for k_R in range(kernel.shape[0]):
                    for k_C in range(kernel.shape[0]):
                        output += kernel[k_R][k_C] * img[i+k_R][j+k_C]

                if model == 1:
                    new_img[i][j] = output/25
                if model != 1:
                    new_img[i][j] = output                    
        return new_img


class Work1(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('AIP+M11102134')
        self.setGeometry(200, 200, 1200, 500)
        
        self.label = QLabel()
        self.label_2 = QLabel()
        
        self.button1 = QPushButton('開啟圖片', self)
        self.button2 = QPushButton('圖片旋轉', self)
        self.button3 = QPushButton('儲存圖片', self)
        self.button4 = QPushButton('卷積運算', self)
        
        self.input_0 = QTextEdit(self)
        self.input_0_0 = QTextEdit(self)
        self.input_1 = QTextEdit(self)
        self.input_2 = QTextEdit(self)
        self.input_3 = QTextEdit(self)
        self.input_4 = QTextEdit(self)
        self.input_5 = QTextEdit(self)
        self.input_6 = QTextEdit(self)
        self.input_7 = QTextEdit(self)
        self.input_8 = QTextEdit(self)
        self.input_9 = QTextEdit(self)
        self.input_10 = QTextEdit(self)
        self.input_11 = QTextEdit(self)
        self.input_12 = QTextEdit(self)
        self.input_13 = QTextEdit(self)
        self.input_14 = QTextEdit(self)
        self.input_15 = QTextEdit(self)
        self.input_16 = QTextEdit(self)
        self.input_17 = QTextEdit(self)
        self.input_18 = QTextEdit(self)
        self.input_19 = QTextEdit(self)
        self.input_20 = QTextEdit(self)
        self.input_21 = QTextEdit(self)
        self.input_22 = QTextEdit(self)
        self.input_23 = QTextEdit(self)
        self.input_24 = QTextEdit(self)
            
        self.button1.clicked.connect(self.Openfile) #繫結QFileDialog
        self.button2.clicked.connect(self.Rotate) #繫結rotate
        self.button3.clicked.connect(self.Save) #繫結Save
        self.button4.clicked.connect(self.Cal) #繫結捲積
                     
        layout = QGridLayout(self)
                      
        layout.addWidget(self.label, 1, 0, 3, 3)
        layout.addWidget(self.label_2, 1, 2, 3, 3)
        layout.addWidget(self.button1, 0, 0, 1, 1)
        layout.addWidget(self.button2, 0, 1, 1, 1)
        layout.addWidget(self.button3, 0, 2, 1, 1)
        layout.addWidget(self.button4, 0, 3, 1, 1)
        
        layout.addWidget(self.input_0,0,4,1,2)
       
        self.input_0_0.setGeometry(QtCore.QRect(840, 45, 60, 20))
        self.input_0.setGeometry(QtCore.QRect(840, 45, 60, 20))
        self.input_1.setGeometry(QtCore.QRect(910, 45, 60, 20))
        self.input_2.setGeometry(QtCore.QRect(980, 45, 60, 20))
        self.input_3.setGeometry(QtCore.QRect(1050, 45, 60, 20))
        self.input_4.setGeometry(QtCore.QRect(1120, 45, 60, 20))
        
        self.input_5.setGeometry(QtCore.QRect(840, 70, 60, 20))
        self.input_6.setGeometry(QtCore.QRect(910, 70, 60, 20))
        self.input_7.setGeometry(QtCore.QRect(980, 70, 60, 20))
        self.input_8.setGeometry(QtCore.QRect(1050, 70, 60, 20))
        self.input_9.setGeometry(QtCore.QRect(1120, 70, 60, 20))
        
        self.input_10.setGeometry(QtCore.QRect(840, 95, 60, 20))
        self.input_11.setGeometry(QtCore.QRect(910, 95, 60, 20))
        self.input_12.setGeometry(QtCore.QRect(980, 95, 60, 20))
        self.input_13.setGeometry(QtCore.QRect(1050, 95, 60, 20))
        self.input_14.setGeometry(QtCore.QRect(1120, 95, 60, 20))
        
        self.input_15.setGeometry(QtCore.QRect(840, 120, 60, 20))
        self.input_16.setGeometry(QtCore.QRect(910, 120, 60, 20))
        self.input_17.setGeometry(QtCore.QRect(980, 120, 60, 20))
        self.input_18.setGeometry(QtCore.QRect(1050, 120, 60, 20))
        self.input_19.setGeometry(QtCore.QRect(1120, 120, 60, 20))
        
        self.input_20.setGeometry(QtCore.QRect(840, 145, 60, 20))
        self.input_21.setGeometry(QtCore.QRect(910, 145, 60, 20))
        self.input_22.setGeometry(QtCore.QRect(980, 145, 60, 20))
        self.input_23.setGeometry(QtCore.QRect(1050, 145, 60, 20))
        self.input_24.setGeometry(QtCore.QRect(1120, 145, 60, 20))
        
        
        
    def Openfile(file):
        filename, _ = QFileDialog.getOpenFileName(file, '開啟檔案', 'Image', '*.jpg *.bmp *.ppm')
        if filename is '':
            return
        file.img = cv2.imread(filename, 1)
        if file.img.size == 1:
            return
        file.Image()
        
    def Image(self):
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_2(self):
        self.img = cv2.imread('out.jpg')
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_3(self):
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Rotate(pict):
        img = pict.img
        (h, w, d) = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        pict.img = cv2.warpAffine(img, M, (w, h))
        pict.Image_3()

        
    def Save(Svfile):
        filename, _ = QFileDialog.getSaveFileName(Svfile, '儲存檔案', 'Image','*.bmp *.jpg *.png')
        if filename is '':
            return
        cv2.imwrite(filename, Svfile.img)

    
    def Cal(self):
        img = self.img
        imgs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰階圖像
        
        try:
            value0 = float(self.input_0_0.toPlainText())
            value1 = float(self.input_1.toPlainText())
            value2 = float(self.input_2.toPlainText())
            value3 = float(self.input_3.toPlainText())
            value4 = float(self.input_4.toPlainText())
            value5 = float(self.input_5.toPlainText())
            value6 = float(self.input_6.toPlainText())
            value7 = float(self.input_7.toPlainText())
            value8 = float(self.input_8.toPlainText())
            value9 = float(self.input_9.toPlainText())  
            value10 = float(self.input_10.toPlainText())  
            value11 = float(self.input_11.toPlainText())   
            value12 = float(self.input_12.toPlainText())   
            value13 = float(self.input_13.toPlainText())    
            value14 = float(self.input_14.toPlainText())    
            value15 = float(self.input_15.toPlainText())   
            value16 = float(self.input_16.toPlainText())   
            value17 = float(self.input_17.toPlainText())   
            value18 = float(self.input_18.toPlainText())    
            value19 = float(self.input_19.toPlainText())    
            value20 = float(self.input_20.toPlainText())   
            value21 = float(self.input_21.toPlainText())   
            value22 = float(self.input_22.toPlainText())   
            value23 = float(self.input_23.toPlainText())   
            value24 = float(self.input_24.toPlainText()) 
            kernel = np.array([
            [value0, value1, value2, value3, value4], 
            [value5, value6, value7, value8, value9], 
            [value10, value11, value12, value13, value14],
            [value15, value16, value17, value18, value19],
            [value20, value21, value22, value23, value24]])
            
            if(int(value8) == 1 & int(value0) == 1 & int(value15) == 1 & int(value3) == 1 & int(value24) == 1):
                dst = conv(imgs,kernel,1)
            else:
                dst = conv(imgs,kernel,0)
        except ValueError as e: 
            
            value0 = 1
            value1 = 1
            value2 = 1
            value3 = 1
            value4 = 1
            value5 = 1
            value6 = 1
            value7 = 1
            value8 = 1
            value9 = 1  
            value10 = 1 
            value11 = 1  
            value12 = 1  
            value13 = 1   
            value14 = 1   
            value15 = 1  
            value16 = 1  
            value17 = 1  
            value18 = 1   
            value19 = 1   
            value20 = 1  
            value21 = 1  
            value22 = 1  
            value23 = 1  
            value24 = 1
            kernel = np.array([
            [value0, value1, value2, value3, value4], 
            [value5, value6, value7, value8, value9], 
            [value10, value11, value12, value13, value14],
            [value15, value16, value17, value18, value19],
            [value20, value21, value22, value23, value24]])
            dst = conv(imgs,kernel,1)
        
        cv2.imwrite('out.jpg',dst)
        self.Image_2()      
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Work1()
    window.show()
    sys.exit(app.exec_())
    


# In[1]:


##################################


# In[2]:


import numpy as np
import cv2 
from matplotlib import pyplot as plt

def conv(img, kernel,model):
    img=np.pad(img,((2,2),(2,2)),'edge')#鏡像邊緣   
    size = img.shape[0]-kernel.shape[0]+1
    new_img = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            output = 0
            for k_R in range(kernel.shape[0]):
                for k_C in range(kernel.shape[0]):
                    output += kernel[k_R][k_C] * img[i+k_R][j+k_C]

            if model == 1:
                new_img[i][j] = output/25
            if model != 1:
                new_img[i][j] = output
            
    return new_img


img = cv2.imread('Noise.jpg')
imgs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰階圖像
conv0 = 1
conv1 = 1
conv2 = 1
conv3 = 1
conv4 = 1
conv5 = 1
conv6 = 1
conv7 = 1
conv8 = 1
conv9 = 1  
conv10 = 1 
conv11 = 1  
conv12 = 1  
conv13 = 1   
conv14 = 1   
conv15 = 1  
conv16 = 1  
conv17 = 1  
conv18= 1   
conv19 = 1   
conv20 = 1  
conv21 = 1  
conv22 = 1  
conv23 = 1  
conv24 = 1
        
kernel = np.array([
[conv0, conv1, conv2, conv3, conv4], 
[conv5, conv6, conv7, conv8, conv9], 
[conv10, conv11, conv12, conv13, conv14],
[conv15, conv16, conv17, conv18, conv19],
[conv20, conv21, conv22, conv23, conv24]])

dst = conv(imgs,kernel,1)
cv2.imwrite('out.jpg',dst)


# In[1]:


import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from numpy.core.fromnumeric import std

def filter(img,kernel):
    img=np.pad(img,((2,2),(2,2)),'edge')
    output = np.zeros (img.shape)
    for row in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            value = kernel * img[(row - 1):(row + 2), (col - 1):(col + 2)]
            output[row-1, col-1] = value.sum ()
    # return output[1:-2,1:-2]
    output = output[1:-2,1:-2]
    # output = (output - output.min())
    # output = output / int(output.max() - output.min())
    # output = output * 255
    # output = np.uint8(output)
    output = 255 - np.abs(output)
    return output

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        
        self.imagepath = ""
        self.img = np.NaN
        self.gray = np.NaN
        self.img_noise = np.NaN
        self.noise = np.NaN
        self.out = np.NaN
        # self.img_num = 0
        # self.noise_std = 0.0

        self.setObjectName("MainWindow")
        self.resize(1030, 642)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setWindowTitle("AIPm10902121")
        
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(10, 10, 70, 20))
        self.btn_load.setObjectName("btn_load")
        self.btn_load.setText("Load")
        self.btn_load.clicked.connect(self.loadfile)
        
        self.conv0_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv0_text.setGeometry(QtCore.QRect(90, 10, 40, 20))
        self.conv0_text.setObjectName("conv0_text")
        self.conv0_text.setEnabled(False)
        self.conv1_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv1_text.setGeometry(QtCore.QRect(140, 10, 40, 20))
        self.conv1_text.setObjectName("conv1_text")
        self.conv1_text.setEnabled(False)
        self.conv2_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv2_text.setGeometry(QtCore.QRect(190, 10, 40, 20))
        self.conv2_text.setObjectName("conv2_text")
        self.conv2_text.setEnabled(False)
        self.conv3_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv3_text.setGeometry(QtCore.QRect(90, 40, 40, 20))
        self.conv3_text.setObjectName("conv3_text")
        self.conv3_text.setEnabled(False)
        self.conv4_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv4_text.setGeometry(QtCore.QRect(140, 40, 40, 20))
        self.conv4_text.setObjectName("conv4_text")
        self.conv4_text.setEnabled(False)
        self.conv5_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv5_text.setGeometry(QtCore.QRect(190, 40, 40, 20))
        self.conv5_text.setObjectName("conv5_text")
        self.conv5_text.setEnabled(False)
        self.conv6_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv6_text.setGeometry(QtCore.QRect(90, 70, 40, 20))
        self.conv6_text.setObjectName("conv6_text")
        self.conv6_text.setEnabled(False)
        self.conv7_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv7_text.setGeometry(QtCore.QRect(140, 70, 40, 20))
        self.conv7_text.setObjectName("conv7_text")
        self.conv7_text.setEnabled(False)
        self.conv8_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv8_text.setGeometry(QtCore.QRect(190, 70, 40, 20))
        self.conv8_text.setObjectName("conv7_text")
        self.conv8_text.setEnabled(False)

        self.btn_set = QtWidgets.QPushButton(self.centralwidget)
        self.btn_set.setGeometry(QtCore.QRect(90, 100, 70, 20))
        self.btn_set.setObjectName("btn_set")
        self.btn_set.setText("Set")
        self.btn_set.setEnabled(False)
        self.btn_set.clicked.connect(self.set)

        self.btn_conv = QtWidgets.QPushButton(self.centralwidget)
        self.btn_conv.setGeometry(QtCore.QRect(240, 10, 70, 20))
        self.btn_conv.setObjectName("btn_conv")
        self.btn_conv.setText("Conv.")
        self.btn_conv.setEnabled(False)
        self.btn_conv.clicked.connect(self.conv)
        
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setGeometry(QtCore.QRect(320, 10, 70, 20))
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setText("Save")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.savefile)
       
        self.btn_exit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_exit.setGeometry(QtCore.QRect(400, 10, 70, 20))
        self.btn_exit.setObjectName("btn_exit")
        self.btn_exit.setText("Exit")
        self.btn_exit.clicked.connect(self.close)        
        
        self.input_label = QtWidgets.QLabel(self.centralwidget)
        self.input_label.setGeometry(QtCore.QRect(10, 120, 47, 12))
        self.input_label.setText("Input:")
        self.input_label.setObjectName("input_label")

        self.output_label = QtWidgets.QLabel(self.centralwidget)
        self.output_label.setGeometry(QtCore.QRect(520, 120, 47, 12))
        self.output_label.setText("Output:")
        self.output_label.setObjectName("output_label")

        self.input_pic = QtWidgets.QLabel(self.centralwidget)
        self.input_pic.setGeometry(QtCore.QRect(10, 132, 500, 500))
        self.input_pic.setText("")
        self.input_pic.setObjectName("input_pic")

        self.output_pic = QtWidgets.QLabel(self.centralwidget)
        self.output_pic.setGeometry(QtCore.QRect(520, 132, 500, 500))
        self.output_pic.setText("")
        self.output_pic.setObjectName("output_pic")

        self.setCentralWidget(self.centralwidget)
        

        
    def loadfile(self):
        
        filePath = QtWidgets.QFileDialog.getOpenFileName(self)
        self.imagepath = filePath[0]
        try:
            self.img = np.NaN
            self.img = cv.imread(self.imagepath, cv.IMREAD_GRAYSCALE)
            cv.imwrite("gray.jpg", self.img)                
            self.input_pic.setPixmap(QtGui.QPixmap('gray.jpg'))
            os.remove('gray.jpg')
            self.conv0_text.setEnabled(True)
            self.conv1_text.setEnabled(True)
            self.conv2_text.setEnabled(True)
            self.conv3_text.setEnabled(True)
            self.conv4_text.setEnabled(True)
            self.conv5_text.setEnabled(True)
            self.conv6_text.setEnabled(True)
            self.conv7_text.setEnabled(True)
            self.conv8_text.setEnabled(True)
            self.btn_set.setEnabled(True)
            self.btn_conv.setEnabled(False)
            
            # self.btn_setstd.setEnabled(True)
            # self.std_text.setEnabled(True)
        except cv.error as e:
            return        
        # self.btn_save.setEnabled(True)
        
    def savefile(self):                  
        save_path = os.path.splitext(self.imagepath)[0]
        cv.imwrite(save_path+'_conv.jpg', self.out)        
        self.btn_save.setEnabled(True)        
        

    def set(self):
        self.conv0_text.setEnabled(False)
        self.conv1_text.setEnabled(False)
        self.conv2_text.setEnabled(False)
        self.conv3_text.setEnabled(False)
        self.conv4_text.setEnabled(False)
        self.conv5_text.setEnabled(False)
        self.conv6_text.setEnabled(False)
        self.conv7_text.setEnabled(False)
        self.conv8_text.setEnabled(False)
        self.btn_conv.setEnabled(True)
        
    def conv(self):
        self.btn_save.setEnabled(True)
        try:
            conv0 = float(self.conv0_text.toPlainText())
            conv1 = float(self.conv1_text.toPlainText())
            conv2 = float(self.conv2_text.toPlainText())
            conv3 = float(self.conv3_text.toPlainText())
            conv4 = float(self.conv4_text.toPlainText())
            conv5 = float(self.conv5_text.toPlainText())
            conv6 = float(self.conv6_text.toPlainText())
            conv7 = float(self.conv7_text.toPlainText())
            conv8 = float(self.conv8_text.toPlainText())
        except ValueError as e:            
            conv0 = 1
            conv1 = 1
            conv2 = 1
            conv3 = 1
            conv4 = 1
            conv5 = 1
            conv6 = 1
            conv7 = 1
            conv8 = 1    
        
        kernel = np.array([
        [conv0, conv1, conv2], 
        [conv3, conv4, conv5], 
        [conv6, conv7, conv8]])
        self.out = filter(self.img, kernel)
        cv.imwrite('out.jpg', self.out)
        self.output_pic.setPixmap(QtGui.QPixmap('out.jpg'))
        os.remove('out.jpg')

        self.conv0_text.setEnabled(True)
        self.conv1_text.setEnabled(True)
        self.conv2_text.setEnabled(True)
        self.conv3_text.setEnabled(True)
        self.conv4_text.setEnabled(True)
        self.conv5_text.setEnabled(True)
        self.conv6_text.setEnabled(True)
        self.conv7_text.setEnabled(True)
        self.conv8_text.setEnabled(True)
        self.btn_conv.setEnabled(False)         

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MyWindow()
    ui.show()
    sys.exit(app.exec_())


# In[ ]:





# In[3]:


import cv2

# Read the original image
img = cv2.imread('bmpImg.bmp') 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:




