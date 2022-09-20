#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        
        self.button1 = QPushButton('開啟圖片', self)
        self.button2 = QPushButton('灰階影像直方圖', self)
        self.button3 = QPushButton('儲存圖片', self)
            
        self.button1.clicked.connect(self.Openfile) #繫結QFileDialog
        self.button2.clicked.connect(self.Histogram) #繫結Histogram
        self.button3.clicked.connect(self.Save) #繫結Save
            
        layout = QGridLayout(self)
        layout.addWidget(self.label, 1, 0, 4, 4)
        layout.addWidget(self.label_2, 1, 4, 4, 4)
        layout.addWidget(self.button1, 0, 0, 1, 1)
        layout.addWidget(self.button2, 0, 1, 1, 1)
        layout.addWidget(self.button3, 0, 2, 1, 1)

        
    def Openfile(file):
        filename, _ = QFileDialog.getOpenFileName(file, '開啟檔案', 'Image', '*.jpg *.bmp *.ppm *.png')
        if filename is '':
            return
        file.img = cv2.imread(filename, 1)
        if file.img.size == 1:
            return
        file.button2.setEnabled(True)
        file.Image()
        
    def Image(self):
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Image_2(self):
        self.img = cv2.imread('Histpct.png')
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QPixmap.fromImage(self.qImg))
          
    def Save(Svfile):
        filename, _ = QFileDialog.getSaveFileName(Svfile, '儲存檔案', 'Image','*.bmp *.jpg *.png')
        if filename is '':
            return
        cv2.imwrite(filename, Svfile.img)
        
    def Histogram(pict):
        img = cv2.cvtColor(pict.img, cv2.COLOR_BGR2GRAY)#轉灰階圖像
        plt.hist(img.ravel(), 256, [0, 256])
        plt.title(" Histogram ")
        plt.xlabel(" Intensity ") #X_Label
        plt.ylabel(" Pixels ") #Y_Label
        plt.savefig('Histpct.png')
        plt.cla()
        pict.button2.setEnabled(False)
        pict.Image_2()


    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Work1()
    window.show()
    sys.exit(app.exec_())
    


# In[ ]:




