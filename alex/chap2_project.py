# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:15:22 2017

@author: alxgr
"""

import pandas     as pd
import matplotlib as plt
import sklearn    as sk
import numpy      as np

######################################
# Data Import
###################################

filepath = '../datasets/housing/housing.csv'
data     = pd.read_csv( filepath )

######################################
# Show data stucture shorcuts
###################################

#data.head()
#data.info()
#data.describe()

#Raw acces to data
#data_array = data.values

# Value count
#data['ocean_proximity'].value_counts()

# Histogram
data.hist( bins = 50 )

# 2D plot
data.plot(kind='scatter',x='longitude',y='latitude')

######################################
# Split data
###################################

train_set, test_set = sk.model_selection.train_test_split( data, test_size = 0.2, random_state = 42)

train_set.hist( bins = 50 )

# Check size
#train_set.values.size
#data.values.size


