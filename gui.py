# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets #importing gui lib
import numpy as np
from keras.preprocessing import image
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout

import cv2
import time
import imutils

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(711, 507)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Browse = QtWidgets.QPushButton(self.centralwidget)
        self.Browse.setGeometry(QtCore.QRect(90, 320, 101, 23))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.Browse.setFont(font)
        self.Browse.setObjectName("Browse")
        self.Camera = QtWidgets.QPushButton(self.centralwidget)
        self.Camera.setGeometry(QtCore.QRect(90, 390, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.Camera.setFont(font)
        self.Camera.setObjectName("Camera")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(90, 440, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.Classify.setFont(font)
        self.Classify.setObjectName("Classify")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(280, 350, 251, 71))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 320, 131, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 20, 421, 20))
        font = QtGui.QFont()
        font.setFamily("AR DESTINE")
        font.setPointSize(24)
        self.label_2.setFont(font)
        self.label_2.setTextFormat(QtCore.Qt.PlainText)
        self.label_2.setObjectName("label_2")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(140, 70, 371, 191))
        self.image.setFrameShape(QtWidgets.QFrame.Box)
        self.image.setText("")
        self.image.setObjectName("image")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.Browse.clicked.connect(self.loadImage)
        
        self.Classify.clicked.connect(self.classifyFunction)
        
        self.Camera.clicked.connect(self.camera) 

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Browse.setText(_translate("MainWindow", " Browse Image"))
        self.Camera.setText(_translate("MainWindow", "Camera"))
        self.Classify.setText(_translate("MainWindow", "classify"))
        self.label.setText(_translate("MainWindow", "Recognized Disease"))
        self.label_2.setText(_translate("MainWindow", "Plant Disease Identifictaion"))
        
    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)")
        if fileName: # If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.image.width(), self.image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.image.setPixmap(pixmap) # Set the pixmap onto the label
            self.image.setAlignment(QtCore.Qt.AlignCenter)
            
    def classifyFunction(self):
        json_file = open('model1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model1.h5")
        classes=["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___healthy",
             "Cherry_(including_sour)___Powdery_mildew","Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___healthy",
             "Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
             "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
             "Potato___healthy","Potato___Late_blight","Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew","Strawberry___healthy","Strawberry___Leaf_scorch",
             "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___healthy","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
             "Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
        path2=self.file
        test_image = image.load_img(path2, target_size = (128, 128))        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        fresult=np.max(result)
        label2=classes[result.argmax()]
        #print(label2)
        self.textEdit.setText(label2)
        
    def camera(self):
        camera = cv2.VideoCapture(0)
        key=input("enter \"q\" to captrue the image")
        if key == "q":
            return_value, cap_image = camera.read()
            cv2.imwrite('opencv'+'.png', cap_image)
        self.file=cap_image
        pixmap = QtGui.QPixmap(cap_image) # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.image.width(), self.image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        self.image.setPixmap(pixmap) # Set the pixmap onto the label
        self.image.setAlignment(QtCore.Qt.AlignCenter)
            
        
        

            
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())