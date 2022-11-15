#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget,QPushButton,QFileDialog,QGridLayout,QLabel)
from PyQt5.QtGui import QImage, QPixmap

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
        
        self.button1 = QPushButton('開啟圖片', self)
        self.button2 = QPushButton('圖片旋轉', self)
        self.button3 = QPushButton('儲存圖片', self)
        self.button4 = QPushButton('直方圖均衡化', self)
            
        self.button1.clicked.connect(self.Openfile) #繫結QFileDialog
        self.button2.clicked.connect(self.Rotate) #繫結rotate
        self.button3.clicked.connect(self.Save) #繫結Save
        self.button4.clicked.connect(self.equalizehist) #繫結直方圖均化
            
        layout = QGridLayout(self)
        layout.addWidget(self.label, 1, 0, 4, 4)
        layout.addWidget(self.label_2, 1, 8, 4, 4)
        layout.addWidget(self.label_3, 1, 4, 4, 4)
        layout.addWidget(self.label_4, 5, 0, 4, 4)
        layout.addWidget(self.label_5, 5, 4, 4, 4)
        layout.addWidget(self.button1, 0, 0, 1, 1)
        layout.addWidget(self.button2, 0, 1, 1, 1)
        layout.addWidget(self.button3, 0, 2, 1, 1)
        layout.addWidget(self.button4, 0, 3, 1, 1)
        
        
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
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_3(self):
        self.img = cv2.imread('enhistorgam.png')
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_3.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_4(self):
        self.img = cv2.imread('historgam_hist.jpg')
        self.img = cv2.resize(self.img, (400, 300), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_4.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_5(self):
        self.img = cv2.imread('enhistorgam_hist.png')
        self.img = cv2.resize(self.img, (400, 300), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_5.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Rotate(pict):
        pict.img = cv2.imread('ori.jpg')
        img = pict.img
        (h, w, d) = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        pict.img = cv2.warpAffine(img, M, (w, h))
        pict.Image_2()
        cv2.imwrite('ori.jpg', pict.img)

        
    def Save(Svfile):
        filename, _ = QFileDialog.getSaveFileName(Svfile, '儲存檔案', 'Image','*.bmp *.jpg *.png')
        if filename is '':
            return
        cv2.imwrite(filename, Svfile.img)
        
    def equalizehist (pict):
        img = cv2.cvtColor(pict.img, cv2.COLOR_BGR2GRAY)
        h,w = img.shape
        hist2 = cv2.calcHist([img], [0],None,[256],[0,256])
        Ary1 = []
        for x in hist2:
            Ary1.append(x[0])
        cv2.imwrite('ori.jpg', img)
        x = [i for i in range(256)]
        plt.figure()
        plt.bar(x, Ary1)
        plt.savefig("historgam_hist.jpg")
        
        H_avg = []
        H_avg.append(Ary1[0])
        for j in range(1, 256):
            H_avg.append(H_avg[j-1] + Ary1[j])   
        min_num = min(Ary1)

        gq = []
        for k in range(0, 256):
            gq.append(round(((H_avg[k] - min_num)/(h*w-min_num))*255))
            
        output = np.zeros(shape=(h,w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                input_img = img[i][j]
                output[i][j] = gq[input_img]
        cv2.imwrite('enhistorgam.png', output)
        pict.Image_3()
        
        enimg = cv2.imread('enhistorgam.png')
        enimg = cv2.cvtColor(enimg, cv2.COLOR_BGR2GRAY)
        hist3 = cv2.calcHist([enimg],[0],None,[256],[0,256])
        Ary2 = []
        for y in hist3:
            Ary2.append(y[0])
        y = [i for i in range(256)]
        plt.figure()
        plt.bar(y, Ary2)
        plt.savefig("enhistorgam_hist.png") 
        pict.Image_4()
        pict.Image_5()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Work1()
    window.show()
    sys.exit(app.exec_())


# In[ ]:




