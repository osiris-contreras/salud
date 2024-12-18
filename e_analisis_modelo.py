import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 

x_train /=255 
x_test /=255

##### cargar modelo  ######

modelo=tf.keras.models.load_model('salidas\\best_model.keras')

####desempeño en evaluación para grupo 1 (Fracturas) #######
prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en evaluación")### conocer el comportamiento de las probabilidades para revisar threshold

prob = modelo.predict(x_test)

# Crear el histograma
sns.histplot(prob, bins=10, kde=False, color="skyblue", edgecolor="black")
plt.title("Probabilidades imágenes en evaluación")
plt.xlabel("Probabilidad")
plt.ylabel("Conteo")

# Calcular los porcentajes para cada barra del histograma
counts, bin_edges = np.histogram(prob, bins=10)
total = len(prob)
for count, edge in zip(counts, bin_edges):
    # Calcular el porcentaje
    percentage = count / total * 100
    # Colocar el texto encima de cada barra
    plt.text(edge + 0.05, count + 5, f'{percentage:.1f}%', ha='center')

plt.show()

threshold_frac=0.30

pred_test=(modelo.predict(x_test)>=threshold_frac).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Fractured', 'No Fractured'])
disp.plot()

### desempeño en evaluación #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en evaluación")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_frac).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Fractured', 'No fractured'])
disp.plot()


########### ##############################################################
####desempeño en evaluación para grupo 2 (No tiene fractura) #######
########### ##############################################################

prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en evaluación")### conocer el comportamiento de las probabilidades para revisar threshold


threshold_no_frac=0.40

pred_test=(modelo.predict(x_test)>=threshold_no_frac).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Fractured', 'No fractured'])
disp.plot()



### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_no_frac).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Fractured', 'No fractured'])
disp.plot()



####### clasificación final ################

prob=modelo.predict(x_test)

clas=['Fractured' if prob >0.8 else 'No Fractured' if prob <0.15 else "Revisión" for prob in prob]

clases, count =np.unique(clas, return_counts=True)

count*100/np.sum(count)