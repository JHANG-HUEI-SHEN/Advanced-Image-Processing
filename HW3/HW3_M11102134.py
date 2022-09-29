#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget,QPushButton,QFileDialog,QGridLayout,QLabel,QInputDialog,QVBoxLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets

class Work1(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        

        self.setWindowTitle('AIP+M11102134')
        self.setGeometry(200, 200, 1000, 500)
        
        self.label = QLabel()
        self.label_2 = QLabel()
        self.label_3 = QLabel()
        self.label_4 = QLabel()
        self.label_5 = QLabel()
        self.label_6 = QLabel()
        self.label_7 = QLabel()
        self.label_8 = QLabel()
        self.label_9 = QLabel()
        self.label_10 = QLabel()
        gridlayout = QGridLayout()
        
        self.button1 = QPushButton('開啟圖片', self)
        self.button2 = QPushButton('影像旋轉', self)
        self.button3 = QPushButton('直方圖', self)
        self.button4 = QPushButton('雜訊產生', self)
         
            
        self.button1.clicked.connect(self.Openfile) #繫結QFileDialog
        self.button2.clicked.connect(self.Rotate) #繫結Histogram
        self.button3.clicked.connect(self.Histogram) #繫結Save
        self.button4.clicked.connect(self.showNumberDialog) #繫結Save
        

            
        layout = QGridLayout(self)
        layout.addWidget(self.label, 1, 0, 4, 4)
        layout.addWidget(self.label_2, 5, 0, 4, 4)
        
        layout.addWidget(self.label_3, 1, 5, 4, 4)
        layout.addWidget(self.label_4, 5, 5, 4, 4)
        
        layout.addWidget(self.label_5, 1, 9, 4, 4)
        layout.addWidget(self.label_6, 5, 9, 4, 4)
        
        layout.addWidget(self.label_7, 1, 13, 4, 4)
        layout.addWidget(self.label_8, 5, 13, 4, 4)
        
        layout.addWidget(self.label_9, 1, 17, 4, 4)
        layout.addWidget(self.label_10, 5, 17, 4, 4)
        
        layout.addWidget(self.button1, 0, 0, 1, 1)
        layout.addWidget(self.button2, 0, 1, 1, 1)
        layout.addWidget(self.button3, 0, 2, 1, 1)
        layout.addWidget(self.button4, 0, 3, 1, 1)

        
    def Openfile(file):
        filename, _ = QFileDialog.getOpenFileName(file, '開啟檔案', 'Image', '*.jpg *.bmp *.ppm *.png')
        if filename is '':
            return
        file.img = cv2.imread(filename, 1)
        if file.img.size == 1:
            return
        file.button2.setEnabled(True)
        file.Image()
        file.button4.setEnabled(True)
        file.button2.setEnabled(False)
        file.button3.setEnabled(False)
        
    def Image(self):
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_2(self):
        self.img = cv2.imread('Histpct.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_3(self):
        self.img = cv2.imread('Gn3.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_3.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_4(self):
        self.img = cv2.imread('Histpct_Gn_noise_4.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_4.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_5(self):
        self.img = cv2.imread('Gn5.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_5.setPixmap(QPixmap.fromImage(self.qImg))
        #self.Image_6()
        
    def Image_6(self):
        self.img = cv2.imread('Histpct_Gn_noise_6.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_6.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_7(self):
        self.img = cv2.imread('Salt_noise.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_7.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_8(self):
        self.img = cv2.imread('Histpct_8.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_8.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_9(self):
        self.img = cv2.imread('Noise.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_9.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_10(self):
        self.img = cv2.imread('Histpct_10.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_10.setPixmap(QPixmap.fromImage(self.qImg))

          
    def Save(Svfile):
        filename, _ = QFileDialog.getSaveFileName(Svfile, '儲存檔案', 'Image','*.bmp *.jpg *.png')
        if filename is '':
            return
        cv2.imwrite(filename, Svfile.img)
        
    def Rotate(pict):
        img = pict.img
        (h, w, d) = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        pict.img = cv2.warpAffine(img, M, (w, h))
        pict.Image_2()
        
        
    def Histogram(pict):
        pict.img = cv2.imread('Org.png')
        img = cv2.cvtColor(pict.img, cv2.COLOR_BGR2GRAY)#轉灰階圖像
        plt.hist(img.ravel(), 256, [0, 256])
        plt.title(" Histogram ")
        plt.xlabel(" Intensity ") #X_Label
        plt.ylabel(" Pixels ") #Y_Label
        plt.savefig('Histpct.png')
        plt.cla()
        pict.button2.setEnabled(False)
        pict.Image_2()
        pict.Image_4()
        pict.Image_6()
        pict.Image_8()
        pict.Image_10()
            
        
    def showNumberDialog(self):
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)#轉灰階圖像
        cv2.imwrite("Org.png", self.img)
        
        qwidget = QWidget()
        data = QInputDialog.getInt(qwidget, "高斯雜訊分佈", "請輸入標準差", 1, 1, 1000, 1)
        data_2 = QInputDialog.getInt(qwidget, "椒鹽雜訊的百分比", "請輸入%，打數字就可", 1, 1, 100, 1)
        prob = (int(data_2[0])/2)*10          
        output = np.zeros(image.shape,np.uint8)
        output_8 = np.zeros(image.shape,np.uint8)

        for i in range(image.shape[0]):               
            for j in range(image.shape[1]):
                randomnum = random.randint(1,1001)   
                if randomnum < prob:           
                    output[i][j] = 0
                    output_8[i][j] = 0  
                elif randomnum > (1000-prob):       
                    output[i][j] = 255
                    output_8[i][j] = 255 
                else:                                   
                    output[i][j] = image[i][j]
                    
        new_im = Image.fromarray(output)
        new_im.save('Noise.png')
        self.Image_9()
        self.button4.setEnabled(False)
        
        amax = np.max(output)
        bmin = np.min(output)
        
        plt.hist(output.ravel(), amax-bmin, [bmin, amax])
        plt.title(" Histogram ")
        plt.xlabel(" Intensity ") #X_Label
        plt.ylabel(" Pixels ") #Y_Label
        plt.savefig('Histpct_10.png')
        plt.cla()
        
        
        
        salt_img = Image.fromarray(output_8)
        salt_img.save('Salt_noise.png')
        self.Image_7()
        
        amax = np.max(output_8)
        bmin = np.min(output_8)
        
        plt.hist(output_8.ravel(), 256, [0, 256])
        plt.title(" Histogram ")
        plt.xlabel(" Intensity ") #X_Label
        plt.ylabel(" Pixels ") #Y_Label
        plt.savefig('Histpct_8.png')
        plt.cla()
        
        #高斯
        self.img = cv2.imread('Org.png')
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)#轉灰階圖像
        if (img.shape[1] % 2) == 1:
            img = np.concatenate([img, img[:,-1:]], axis=1)
        Gn_output = np.zeros(img.shape,np.uint8)
        Gn_output_3 = np.zeros(img.shape,dtype=float)
        
        for i in range(0,img.shape[0]):               
            for j in range(0,img.shape[1]-1,2):
                
                phi = round(random.uniform(0,1),7) #Φ
                r = round(random.uniform(0,1),7) 
                
                s = np.sqrt(-2*np.log(r))
                cl = 2*phi*np.pi
                
                z1 = s*int(data[0]) * np.cos(cl)
                z2 = s*int(data[0]) * np.sin(cl)
                
                k1 = img[i,j]        
                k2 = img[i,j+1]
                f1 = z1 + k1
                f2 = z2 + k2
                
                
                
                if f1 < 0:
                    Gn_output[i,j] = 0
                    z1 = -k1
                elif f1 > 255:
                    Gn_output[i,j] = 255
                    z1 = 255-k1
                else:
                    Gn_output[i,j] = np.uint8(f1)
                    
          
                if f2 < 0:
                    Gn_output[i,j+1] = 0
                    z2 = -k2
                elif f2 > 255:
                    Gn_output[i,j+1] = 255
                    z2 = 255-k2
                else:
                    Gn_output[i,j+1] = np.uint8(f2)
                    
                Gn_output_3[i,j] = z1   
                Gn_output_3[i,j+1] =  z2
                    
                
        new_Gn_im = Image.fromarray(Gn_output)
        new_Gn_im.save('Gn5.png')
        self.Image_5()
        
        b=np.max(Gn_output)
        a=np.min(Gn_output)
        
        plt.hist(Gn_output.ravel(), b-a, [a, b])
        plt.title(" Histogram ")
        plt.xlabel(" Intensity ") #X_Label
        plt.ylabel(" Pixels ") #Y_Label
        plt.savefig('Histpct_Gn_noise_6.png')
        plt.cla()
        
        
                    
        imgs = cv2.imread('Org.png')
        imgg = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)#轉灰階圖像
        Gn_output_3_img = np.uint8(Gn_output_3)
        new_Gn_im = Image.fromarray(Gn_output_3_img)
        new_Gn_im.save('Gn3.png')
        
        b=int(np.max(Gn_output_3))
        a=int(np.min(Gn_output_3))
        
        
        plt.hist(Gn_output_3.ravel(), b-a, [a, -a])
        plt.title(" Histogram ")
        plt.xlabel(" Intensity ") #X_Label
        plt.ylabel(" Pixels ") #Y_Label
        plt.savefig('Histpct_Gn_noise_4.png')
        plt.cla()
        
        self.Image_3()
        self.button3.setEnabled(True)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Work1()
    window.show()
    sys.exit(app.exec_())
    


# In[ ]:




