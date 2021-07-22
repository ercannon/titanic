#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 22:18:54 2021

@author: cannon
"""

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

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