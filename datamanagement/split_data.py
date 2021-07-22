#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 22:12:51 2021

@author: cannon
"""
from sklearn.model_selection import train_test_split

def split_data(data):
    return train_test_split(data, test_size=0.2, random_state=42)