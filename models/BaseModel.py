#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:29:55 2021

@author: cannon
"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

class BaseModel:
    def __init__(self,_model):
        self.model=_model
        
    def confusion_matrix(self,train_data,train_labels):
        y_train_pred = cross_val_predict(self.model, train_data, train_labels, cv=3)
        print(confusion_matrix(train_labels, y_train_pred))