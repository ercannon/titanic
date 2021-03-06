#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 22:18:54 2021

@author: cannon
"""

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

def clean_data(data):
    #quitamos el número de cabina, hay valores a vacío. Nombre no aporta nada
    titanic_numeric=data.drop(["Cabin","Name","Sex","Ticket","Embarked","PassengerId"],axis=1)
    #hay valores de Age a vacío, podemos ponerles la mediana.
    #primero me quedo solo con los valores vacios
    imputer = SimpleImputer(strategy="median")
    imputer.fit(titanic_numeric)
    X=imputer.transform(titanic_numeric)
    titanic_transformed=pd.DataFrame(X,columns=titanic_numeric.columns)
    data["Sex"] = np.where((data.Sex == 'male'),1,0)
    print(data["Sex"])

def clean_data_manually(data):
    #quito columnas que no valen para nada
    data_clean=data.drop(["Name","PassengerId","Ticket","Embarked"],axis=1)
    #si camarote es na suopngo que no tiene, así que le meto un 0, y los otros a 1
    data_clean["Cabin"]=data_clean["Cabin"].fillna(0)
    data_clean["Cabin"] = np.where((data_clean.Cabin == 0),0,1)
    #limpio el na de Embarked
    #data_clean["Embarked"]=data_clean["Embarked"].fillna("S")
    median = data_clean["Fare"].median()
    data_clean["Fare"]=data_clean["Fare"].fillna(median)
    median = data_clean["Age"].median()
    data_clean["Age"]=data_clean["Age"].fillna(median)

    data_clean["Sex"] = np.where((data_clean.Sex == "male"),0,1)
    data_clean["Sex"].fillna(0)
    
    return data_clean
    
    
def clean_data_pipelines(data):
     #quitamos el número de cabina, hay valores a vacío. Nombre no aporta nada
     titanic_numeric=data.drop(["Name","Sex","Ticket","Embarked","PassengerId","Pclass"],axis=1)
     #si camarote es na suopngo que no tiene, así que le meto un 0, y los otros a 1
     data["Cabin"]=data["Cabin"].fillna(0)
     data["Cabin"] = np.where((data.Cabin == 0),0,1)
     #limpio el na de Embarked
     data["Embarked"]=data["Embarked"].fillna("S")
     #Mediana a los valores numeros que tienen Na y luego standarizo los valores
     num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])
     #titanic_num_tr = num_pipeline.fit_transform(titanic_numeric)
     num_attribs = list(titanic_numeric)
     cat_attribs = ["Sex","Pclass","Embarked"]
     full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OrdinalEncoder(), cat_attribs),
        ])
     titanic_prepared = full_pipeline.fit_transform(data)
     return titanic_prepared