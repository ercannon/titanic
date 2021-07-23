#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:36:20 2021

@author: cannon
"""
from datamanagement.analyze_data import load_titanic_data,analyze,save_prediction
from datamanagement.split_data import split_data
from datamanagement.data_clean import clean_data_pipelines
from models.Model import Model
from constants import data_constants as const

modelobj = Model()

model_name = const.BEST_MODEL_ID

if (not modelobj.load_model(model_name)):
    data=load_titanic_data()
    #analyze(data)
    train_set, test_set = split_data(data)
    train_data,train_labels=train_set.drop("Survived",axis=1),train_set["Survived"]
    #analyze(train_set)
    train_data_cleaned=clean_data_pipelines(train_data)
    print("datos limpiados")
    #modelobj.estadisticas(train_data_cleaned,train_labels)
    #modelobj.validacion_cruzada(train_data_cleaned, train_labels)
    modelobj.fine_tuning(train_data_cleaned,train_labels)
    print("fine tuning done")
    #modelobj.validacion_cruzada(train_data_cleaned, train_labels)
    modelobj.train(train_data_cleaned,train_labels)
    print("modelo entrenado")
    test_data,test_labels=test_set.drop("Survived",axis=1),test_set["Survived"]
    test_data_cleaned=clean_data_pipelines(test_data)
    prediction=modelobj.predict(test_data_cleaned)
    modelobj.print_estadisticas(test_labels, prediction)
    modelobj.save_model(model_name)
#sacamos los resultados finales
print("Empeanzo la prediccion")
prediction_dataset = load_titanic_data(train_flag=False)
final_data_prepared = clean_data_pipelines(prediction_dataset)
final_prediction = modelobj.predict(final_data_prepared)
print("antes de salvar")
save_prediction(final_prediction,prediction_dataset["PassengerId"])
print("Finalizado")

