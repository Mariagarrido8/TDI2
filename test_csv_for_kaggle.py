#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
% TDImagen PROYECTO 2: CLASIFICACIÓN DE IMÁGENES                        %
% Creación de CSV para subir la solución a Kaggle                       %
%                                                                       %
% --------------------------------------------------------------------- %
%                                                                       %
% Created on Thu Apr 16 12:41:19 2020                                   %
% @author: Miguel-Angel Fernandez-Torres                                %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
import os, csv
from project2 import *

def test_prediction_csv(dir_images_train_name='database/train', 
                        y_train_root = 'database/y_train.npy',
                        dir_images_test_name='database/test', 
                        csv_name='test_prediction_gray_canny_hough_entropy_glcm.csv'):
    # Llamadas a las funciones que implementan el sistema de clasificación
    # de imágenes.
    # <EXTRACCIÓN DE CARACTERÍSTICAS DE IMÁGENES DE ENTRENAMIENTO Y TEST>
    print('Extrayendo características y etiquetas del conjunto de entrenamiento...')
    X_train = databaseFeatures(database_set_root=dir_images_train_name)
    y_train = np.load(y_train_root) # <VECTOR DE ETIQUETAS DE ENTRENAMIENTO>
    print('Extrayendo características del conjunto de test...')
    X_test = databaseFeatures(database_set_root=dir_images_test_name)
    
    # <CLASIFICADOR>
    # - FASE DE ENTRENAMIENTO
    print('Entrenando el clasificador...')
    scaler, model = train_classifier(X_train, y_train)
    # - FASE DE TEST
    print('Obteniendo predicciones para el conjunto de test...')
    y_pred = test_classifier(scaler, model, X_test)
    
    # CREACIÓN DE CSV PARA KAGGLE
    print('Creando CSV para Kaggle...')
    test_images_roots = np.sort([name for name in os.listdir(dir_images_test_name) if name.endswith('.png')])
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Predicted"])
        for i in np.arange(np.size(test_images_roots)):        
            # - - - Escritura de predicciones en fichero .csv
            writer.writerow([test_images_roots[i][:-4], y_pred[i]])
            print('Predicción de imagen '+str(i)+' añadida a CSV..')
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print('¡CSV creado!')
            
if __name__ == '__main__':
    # Creación de CSV, asumiendo parámetros por defecto.
    test_prediction_csv()
    
#from sklearn.model_selection import train_test_split
#
#X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.30, random_state=64)
#
#scaler, model = train_classifier(X_train, y_train)
#
#y_pred = test_classifier(scaler, model, X_val)
# 
#auc_score = eval_classifier(y_val, y_pred)