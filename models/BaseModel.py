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

def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.show()

class BaseModel:
    def __init__(self,_model):
        self.model=_model
  
    def estadisticas(self,train_data,train_labels):
        print("Accuracy ratio :",cross_val_score(self.model, train_data, train_labels, cv=3, scoring="accuracy"))
        y_train_pred = cross_val_predict(self.model, train_data, train_labels, cv=3)
        print("Matriz de confusion",confusion_matrix(train_labels, y_train_pred))
        print("Precision: ",precision_score(train_labels, y_train_pred))
        print("Recall ",recall_score(train_labels, y_train_pred))
        print("F1 score ",f1_score(train_labels, y_train_pred))
        fpr, tpr, thresholds = roc_curve(train_labels, y_train_pred)
        plot_roc_curve(fpr, tpr)
    
    
