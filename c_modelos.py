#######################################################################
######################## Anallítica en Salud ##########################
############################## Equipo 4 ###############################
############################### Modelos################################
#######################################################################

import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo
import pandas as pd

from sklearn import tree


import cv2 ### para leer imagenes jpeg
### pip install opencv-python

from matplotlib import pyplot as plt #

### cargar bases_procesadas ####

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

############################################################
################ Preprocesamiento ##############
############################################################

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=255 ### escalaro para que quede entre 0 y 1
x_test /=255


###### verificar tamaños

x_train.shape
x_test.shape

np.product(x_train[1].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

##### convertir a 1 d array ############
x_train2=x_train.reshape(8863,150528)
x_test2=x_test.reshape(600, 150528)
x_train2.shape
x_test2.shape

x_train2[1]


############################################################
###############Seleccionar un indicador ####################
############################################################

## * Sensibilidad (Recall): Indica el porcentaje de fracturas correctamente identificadas. 
# Es crucial en radiología, ya que los falsos negativos (fracturas no detectadas) podrían retrasar un tratamiento necesario. 
# La sensibilidad debería ser alta para reducir el riesgo de pasar por alto casos críticos.

## * Especificidad: Mide el porcentaje de imágenes "No fracturadas" correctamente clasificadas. 
# Aunque es importante, suele tener menor prioridad en este tipo de problemas comparado con la sensibilidad.

## * F1 Score: Equilibra precisión y sensibilidad, lo cual es útil si buscamos un solo indicador que capture ambos aspectos.
#  Sin embargo, podemos priorizar la sensibilidad si la detección de fracturas es más crítica.

## * AUC-ROC (Área Bajo la Curva): Es una medida completa del rendimiento del modelo en diferentes umbrales de clasificación. 
# Permite evaluar tanto la sensibilidad como la especificidad y es útil para comparar modelos.

## * Accuracy representa la proporción de imágenes que el modelo clasifica correctamente,
#  tanto "Fracturadas" como "No fracturadas". 

############Analisis problema ###########
# --  AUC-ROC: El AUC-ROC es una métrica útil en este contexto porque evalúa la capacidad del modelo para distinguir
# entre imágenes "Fracturadas" y "No fracturadas" en una variedad de umbrales de decisión. Este indicador permite
# un análisis equilibrado entre sensibilidad (capacidad de detectar fracturas) y especificidad (capacidad de evitar
# falsos positivos) en función de diferentes puntos de corte. Un alto valor de AUC sugiere que el modelo es capaz
# de discriminar correctamente entre las dos clases, lo cual es crucial en contextos médicos donde es importante
# minimizar tanto los falsos negativos como los falsos positivos. Además, el AUC-ROC facilita la comparación entre
# modelos y permite ajustar el umbral de decisión según la prioridad clínica, maximizando la sensibilidad sin
# comprometer excesivamente la especificidad.
# -- Recall (Sensibilidad): El Recall es una métrica importante en este contexto médico porque mide la capacidad del
# modelo para detectar correctamente las imágenes "Fracturadas" y minimizar los falsos negativos, es decir, los casos
# de fracturas que podrían no ser detectados. Aunque el AUC-ROC proporciona una visión completa del balance entre
# sensibilidad y especificidad, el Recall sigue siendo esencial como métrica secundaria para asegurar que el modelo
# mantiene una alta capacidad de detección de fracturas en su configuración final. Esto ayuda a garantizar que se
# prioricen los casos críticos y se reduzca el riesgo de pasar por alto fracturas que requieren atención urgente.
# -- Especificidad: Complementaria para reducir falsos positivos y evitar que se clasifiquen erróneamente casos 
# sin fractura como fracturados.
# -- Accuracy y F1 Score: Indicadores secundarios; aunque útiles, no son tan críticos en un contexto médico con 
# clases potencialmente desbalanceadas.


############################################################
################ Probar modelos de tradicionales############
############################################################

################################## RandomForest ##########################################
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Entrenamiento del modelo
rf = RandomForestClassifier()
rf.fit(x_train2, y_train)

# Predicciones y evaluación en el conjunto de entrenamiento
pred_train = rf.predict(x_train2)
print("Entrenamiento - Reporte de Clasificación:")
print(classification_report(y_train, pred_train))

train_auc = roc_auc_score(y_train, pred_train)
print(f"AUC-ROC Entrenamiento: {train_auc:.2f}")

# Predicciones y evaluación en el conjunto de prueba
pred_test = rf.predict(x_test2)
print("Prueba - Reporte de Clasificación:")
print(classification_report(y_test, pred_test))

test_auc = roc_auc_score(y_test, pred_test)
print(f"AUC-ROC Prueba: {test_auc:.2f}")

# Matriz de Confusión en conjunto de prueba
cm = confusion_matrix(y_test, pred_test)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)  # Cálculo manual de especificidad
recall = tp / (tp + fn)  # Cálculo de sensibilidad (recall)

print("Matriz de Confusión (Prueba):")
print(cm)
print(f"Sensibilidad (Recall): {recall:.2f}")
print(f"Especificidad: {specificity:.2f}")

# Conclusión del modelo Random Forest:
# El modelo Random Forest mostró sobreajuste, con un desempeño perfecto en entrenamiento pero bajo en prueba 
# (AUC-ROC 0.74, baja sensibilidad). Se probarán otros modelos para mejorar la generalización.


################################# Decision Tree ########################################
from sklearn import tree, metrics

# Entrenamiento del modelo
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(x_train2, y_train)

# Evaluación en el conjunto de entrenamiento
pred_train = clf_dt.predict(x_train2)
print("Entrenamiento - Reporte de Clasificación:")
print(metrics.classification_report(y_train, pred_train))

train_auc = metrics.roc_auc_score(y_train, pred_train)
print(f"AUC-ROC Entrenamiento: {train_auc:.2f}")

# Evaluación en el conjunto de prueba
pred_test = clf_dt.predict(x_test2)
print("Prueba - Reporte de Clasificación:")
print(metrics.classification_report(y_test, pred_test))

test_auc = metrics.roc_auc_score(y_test, pred_test)
print(f"AUC-ROC Prueba: {test_auc:.2f}")

# Conclusión del modelo Decision Tree:
# El modelo muestra un fuerte sobreajuste, con desempeño perfecto en entrenamiento pero bajo en prueba (AUC-ROC 0.52),
# lo que indica que no generaliza bien. Se probarán otros modelos.


############################################################
################ Probar modelos de redes neuronales ########
############################################################
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping

# Definición del modelo de red neuronal
fc_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Configuración del optimizador y la función de pérdida
fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Recall', 'Precision'])

# Early stopping para detener entrenamiento si no hay mejoras
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento del modelo
fc_model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Evaluación en el conjunto de prueba
test_loss, test_acc, test_auc, test_recall, test_precision = fc_model.evaluate(x_test, y_test, verbose=2)
print(f"Test AUC: {test_auc:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test Precision: {test_precision:.2f}")

# Matriz de confusión y reporte de clasificación en el conjunto de prueba
pred_test = (fc_model.predict(x_test) > 0.50).astype('int')
cm = metrics.confusion_matrix(y_test, pred_test, labels=[1, 0])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fracturado', 'No Fracturado'])
disp.plot()

print("Reporte de Clasificación - Prueba:")
print(metrics.classification_report(y_test, pred_test))

# Conclusión del modelo de red neuronal:
# El modelo de red neuronal muestra una precisión y sensibilidad moderadas en el conjunto de prueba (AUC 0.59, Recall 0.59),
# indicando dificultades para generalizar. La arquitectura actual no captura suficientemente bien los patrones de fractura.
# haremos mejoras. 

##############################################mejoras####################################

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten

# Modelo de red neuronal mejorado
fc_model2 = tf.keras.models.Sequential([
    Flatten(input_shape=x_train.shape[1:]),  # Aplanar la entrada
    Dense(256, activation='relu'),  # Capa oculta más grande
    BatchNormalization(),  # Normalización para estabilizar el entrenamiento
    Dropout(0.5),  # Regularización para evitar overfitting
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Capa de salida
])

# Configuración del modelo
fc_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'AUC', 'Recall', 'Precision'])

# Early stopping para detener el entrenamiento si no hay mejoras
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento del modelo
history = fc_model2.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Evaluación en el conjunto de prueba
test_loss, test_acc, test_auc, test_recall, test_precision = fc_model2.evaluate(x_test, y_test, verbose=2)
print(f"Test AUC: {test_auc:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test Precision: {test_precision:.2f}")

# Matriz de confusión y reporte de clasificación
pred_test = (fc_model2.predict(x_test) > 0.50).astype('int')
cm = metrics.confusion_matrix(y_test, pred_test, labels=[1, 0])
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fractured', 'No Fractured'])
disp.plot()

print("Reporte de Clasificación - Prueba:")
print(metrics.classification_report(y_test, pred_test))

##La red neuronal mejorada mostró un buen AUC (0.94), indicando capacidad para discriminar entre clases,
#  pero presenta un desequilibrio notable en las predicciones: alta precisión (0.98) pero baja sensibilidad (0.41). 
# Esto implica que el modelo identifica bien los casos "No Fracturados", pero falla en detectar casos "Fracturados",
#  lo cual es crítico para nuestro contexto.