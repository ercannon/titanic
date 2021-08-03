#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 22:02:04 2021

@author: cannon
"""
from datamanagement.analyze_data import load_titanic_data,analyze
from datamanagement.split_data import split_data
from models.Model import Model
from datamanagement.data_clean import clean_data_pipelines,clean_data_manually


modelobj = Model()

data=load_titanic_data()
analyze(data)

# =============================================================================
# indexNames = data[ data['Fare'] == 0 ].index
# data.drop(indexNames , inplace=True)
# data.reset_index(inplace=True,drop=True)
# =============================================================================
    
train_set, test_set = split_data(data)
train_data,train_labels=train_set.drop("Survived",axis=1),train_set["Survived"]
analyze(train_set)
#train_data_cleaned=clean_data_pipelines(train_data)
train_data_cleaned=clean_data_manually(train_data)
modelobj.validacion_cruzada(train_data_cleaned, train_labels)


