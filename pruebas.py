#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:25:31 2020

@author: mpostigo
"""

import os
import numpy as np
from skimage import io, filters, color
from project2 import *
import matplotlib.pyplot as plt

data_dir= os.curdir+'/database'

#Cargamos las imágenes
train_imgs_files = [ os.path.join(data_dir,'train',f) for f in sorted(os.listdir(os.path.join(data_dir,'train'))) 
            if (os.path.isfile(os.path.join(data_dir,'train',f)) and f.endswith('.png')) ]

test_imgs_files = [ os.path.join(data_dir,'test',f) for f in sorted(os.listdir(os.path.join(data_dir,'test'))) 
            if (os.path.isfile(os.path.join(data_dir,'test',f)) and f.endswith('.png')) ]

train_imgs_files.sort()
test_imgs_files.sort()
print("Número de imágenes de train", len(train_imgs_files))
print("Número de imágenes de test", len(test_imgs_files))

# Cargamos los labels
#'y_train.npy': 0="habitación ordenada", 1="habitación desordenada"
Y = np.load(data_dir+'/y_train.npy')
train_images = io.imread_collection(train_imgs_files)
test_images = io.imread_collection(test_imgs_files)
my_dict = {0:'ordenada', 1:'desordenada'}

plt.imshow(train_images[130])
plt.title("Imagen original: habitacion %s"%(my_dict.get(Y[130])))


#1. IMAGE PROCESSING
processed_image = imageProcessing(train_images[130])

#plt.figure(figsize=(10,20))
#plt.subplot(1,2,1)
#plt.imshow(processed_image['image'])
#plt.title('Original')
#plt.subplot(1,2,2)
#plt.imshow(processed_image['image_gray'], cmap='gray')
#plt.title('Gray')
#plt.subplot(1,4,3)
#plt.imshow(processed_image['edge'], cmap='gray')
#plt.title('Edge')
#plt.subplot(1,4,4)
#plt.imshow(processed_image['mask'], cmap='gray')
#plt.title('Mask')

#Extracción de caracteristicas
features = extractFeatures(processed_image)

#Matriz de caracteristicas
X = databaseFeatures(database_set_root="database/train")

#Entrenar el clasificador
scaler, model = train_classifier(X, Y)

#Predecimos xtest con el modelo entrenado 
X_test = databaseFeatures(database_set_root="database/test")
y_pred = test_classifier(scaler, model, X_test)

for i in range (len(test_images)):
    plt.imshow(test_images[i])
    plt.title("Desordenada %.4f"%(y_pred[i]))
    if (i==5):
        break
    plt.show()

#Conjunto de validación  (no es para csv)
    
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.30, random_state=64)

scaler, model = train_classifier(X_train, y_train)

y_pred = test_classifier(scaler, model, X_val)
 
auc_score = eval_classifier(y_val, y_pred)