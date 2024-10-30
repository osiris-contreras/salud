import numpy as np
#pip install tqdm

import cv2 ### para leer imagenes jpeg
#pip install opencv-python


import sys
import os
path='C:/cod/salud'
os.chdir(path) ### definir directorio de trabajo
sys.path.append(path+'/')  ###setting path to read user defined functions


from matplotlib import pyplot as plt ## para gráfciar imágnes
import a_funciones as fn #### funciones personalizadas, carga de imágenes
import joblib ### para descargar array

### Exploracion con algunas imagenes

img1=cv2.imread('data\\val\\not fractured\\1.jpg')
img1_1=cv2.imread('data\\val\\not fractured\\1-rotated1.jpg')

img2=cv2.imread('data\\train\\fractured\\10.jpg')
img2_2=cv2.imread('data\\train\\fractured\\10-rotated1.jpg')
plt.imshow(img1)
plt.title('Not Fractured')
plt.show()

plt.imshow(img1_1)
plt.title('Not Fractured')
plt.show()

plt.imshow(img2)
plt.title('Fractured')
plt.show()

plt.imshow(img2_2)
plt.title('Fractured')
plt.show()

img1.shape
img1_1.shape
img2.shape
img2_2.shape


np.prod(img2.shape)

img2_r = cv2.resize(img2 ,(224,224))
plt.imshow(img2_r)
plt.title('Fractured')
plt.show()
np.prod(img2_r.shape)


################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################


width = 224 #tamaño para reescalar imágen al valor comun
num_classes = 2 #clases variable respuesta
trainpath = 'data/train/'
testpath = 'data/val/'

x_train, y_train, file_list = fn.img2data(trainpath, width) #Run in train
x_test, y_test, file_list = fn.img2data(testpath, width) #Run in test




#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train.shape
x_test.shape


np.prod(x_train[1].shape)
y_train.shape


x_test.shape
y_test.shape

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "salidas\\x_train.pkl")
joblib.dump(y_train, "salidas\\y_train.pkl")
joblib.dump(x_test, "salidas\\x_test.pkl")
joblib.dump(y_test, "salidas\\y_test.pkl")