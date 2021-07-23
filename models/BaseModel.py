#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:29:55 2021

@author: cannon
"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.show()

class BaseModel:
    def __init__(self,_model):
        self.model=_model
  
    def mean_squared_error(self,labels,predictions):
        lin_mse = mean_squared_error(labels, predictions)
        return np.sqrt(lin_mse)
  
    def validacion_cruzada(self,data,labels):
        print("Accuracy ratio :",cross_val_score(self.model, data, labels, cv=3, scoring="accuracy"))
        y_train_pred = cross_val_predict(self.model, data, labels, cv=3)
        self.print_estadisticas(labels,y_train_pred)
  
    def print_estadisticas(self,labels,predictions):
        print("Matriz de confusion",confusion_matrix(labels, predictions))
        print("Precision: ",precision_score(labels, predictions))
        print("Recall ",recall_score(labels, predictions))
        print("F1 score ",f1_score(labels, predictions))
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        plot_roc_curve(fpr, tpr)
  
    #deprecated
    def estadisticas(self,data,labels):
        print("Accuracy ratio :",cross_val_score(self.model, data, labels, cv=3, scoring="accuracy"))
        y_train_pred = cross_val_predict(self.model, data, labels, cv=3)
        print("Matriz de confusion",confusion_matrix(labels, y_train_pred))
        print("Precision: ",precision_score(labels, y_train_pred))
        print("Recall ",recall_score(labels, y_train_pred))
        print("F1 score ",f1_score(labels, y_train_pred))
        fpr, tpr, thresholds = roc_curve(labels, y_train_pred)
        plot_roc_curve(fpr, tpr)
    
    def train(self,data,labels):
        self.model.fit(data,labels)
    
    def predict(self, data):
        return self.model.predict(data)
        
    
