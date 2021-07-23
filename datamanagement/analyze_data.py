#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:32:34 2021

@author: cannon
"""

import os
from constants import data_constants as const
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

#pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None, "display.max_columns", None)

def load_titanic_data(housing_path=const.TITANIC_PATH,train_flag=True):
    if train_flag:
        filename = "train.csv"
    else:
        filename = "test.csv"
    csv_path = os.path.join(housing_path, filename)
    return pd.read_csv(csv_path)

def save_prediction(prediction,ids):
    data= pd.DataFrame([ids,prediction]).T
    data.columns = ['PassengerId','Survived']
    data.to_csv("results.csv", index=False)
    return data

def analyze(titanic):
    print(titanic.head())
    print(titanic.info())
    print(titanic.describe())
    titanic.hist(bins=50, figsize=(20,15))
    plt.show()
    #Como es pequenyo, busco la matriz de correlacin. veo que la tarifa influye
    corr_matrix = titanic.corr()
    print("Correlation matrix ",corr_matrix["Survived"])
    #vamos a pintar los valores más "parecidos"
    attributes = ["Survived", "Fare", "Parch"]
    scatter_matrix(titanic[attributes], figsize=(12, 8))
    titanic.plot(kind="scatter", x="Fare", y="Survived",
                 alpha=0.1)


