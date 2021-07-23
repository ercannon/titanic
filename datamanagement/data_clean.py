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
    
def clean_data_pipelines(data):
     #quitamos el número de cabina, hay valores a vacío. Nombre no aporta nada
     titanic_numeric=data.drop(["Cabin","Name","Sex","Ticket","Embarked","PassengerId","Pclass"],axis=1)
     #Mediana a los valores numeros que tienen Na y luego standarizo los valores
     num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])
     #titanic_num_tr = num_pipeline.fit_transform(titanic_numeric)
     num_attribs = list(titanic_numeric)
     cat_attribs = ["Sex"]
     full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OrdinalEncoder(), cat_attribs),
        ])
     titanic_prepared = full_pipeline.fit_transform(data)
     return titanic_prepared