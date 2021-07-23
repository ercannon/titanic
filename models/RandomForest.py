#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:15:11 2021

@author: cannon
"""
from sklearn.ensemble import RandomForestClassifier

from models.BaseModel import BaseModel

class Model(BaseModel):
    def __init__(self):
        super().__init__(RandomForestClassifier(random_state=42))

    
    
        