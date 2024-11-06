import numpy as np
import pandas as pd
import cv2 ### para leer imagenes jpeg
### pip install opencv-python
import a_funciones as fn#### funciones personalizadas, carga de imágenes
import tensorflow as tf
import openpyxl

import sys
import os

ruta =os.getcwd()
sys.path.append(ruta)
width = 224 #tamaño para reescalar imágen al valor comun

if __name__=="__main__":

    #### cargar datos ####
    ##path = (f'{ruta}\\data\\despliegue\\')
    path = 'data/despliegue/'
    x, _, files= fn.img2data(path,width) #cargar datos de despliegue

    x=np.array(x) ##imagenes a predecir

    x=x.astype('float')######convertir para escalar
    x/=255######escalar datos


    files2= [name.rsplit('.', 1)[0] for name in files] ### eliminar extension a nombre de archivo

    modelo=tf.keras.models.load_model(f'{ruta}\\salidas\\best_model.keras') ### cargar modelo
    prob=modelo.predict(x)


    clas=['Fractured' if prob >0.8 else 'No Fractured' if prob <0.15 else "Revisión" for prob in prob]

    res_dict={
        "paciente": files2,
        "clas": clas   
    }
    resultados=pd.DataFrame(res_dict)

resultados.to_excel(f'{ruta}\\salidas\\clasificados.xlsx', index=False)