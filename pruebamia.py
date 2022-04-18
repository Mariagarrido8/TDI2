# -*- coding: utf-8 -*-
"""
Created on Tue May 12 01:34:40 2020

@author: María
"""

import os
import numpy as np
from numpy.fft import fftshift
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, feature
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt


img_root = './database/train/040.png'

image = io.imread(img_root)
image_gray = color.rgb2gray(image)
edge = feature.canny(image_gray, sigma=2)

lines = probabilistic_hough_line(edge)
plt.figure()
plt.imshow(lines)
plt.imshow()
