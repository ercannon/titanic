#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:32:34 2021

@author: cannon
"""

import os
from constants import data_constants as const
import pandas as pd

def load_titanic_data(housing_path=const.TITANIC_PATH):
    csv_path = os.path.join(housing_path, "train.csv")
    return pd.read_csv(csv_path)