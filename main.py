#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:36:20 2021

@author: cannon
"""
from datamanagement.analyze_data import load_titanic_data,analyze
from datamanagement.split_data import split_data
from datamanagement.data_clean import clean_data_pipelines
from models.Model import Model

data=load_titanic_data()
#analyze(data)
train_set, test_set = split_data(data)
train_data,train_labels=train_set.drop("Survived",axis=1),train_set["Survived"]
#analyze(train_set)
train_data_cleaned=clean_data_pipelines(train_data)
modelobj = Model()
#modelobj.estadisticas(train_data_cleaned,train_labels)
#modelobj.validacion_cruzada(train_data_cleaned, train_labels)
modelobj.train(train_data_cleaned,train_labels)
test_data,test_labels=test_set.drop("Survived",axis=1),test_set["Survived"]
test_data_cleaned=clean_data_pipelines(test_data)
prediction=modelobj.predict(test_data_cleaned)
modelobj.print_estadisticas(test_labels, prediction)
