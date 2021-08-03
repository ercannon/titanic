#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 22:02:04 2021

@author: cannon
"""
from datamanagement.analyze_data import load_titanic_data,analyze
from datamanagement.split_data import split_data
from models.Model import Model
from datamanagement.data_clean import clean_data_pipelines


modelobj = Model()

data=load_titanic_data()
analyze(data)
train_set, test_set = split_data(data)
train_data,train_labels=train_set.drop("Survived",axis=1),train_set["Survived"]
analyze(train_set)
train_data_cleaned=clean_data_pipelines(train_data)
modelobj.estadisticas(train_data_cleaned,train_labels)



