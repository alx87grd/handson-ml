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
#data.hist( bins = 50 )

# Correlation coef.
data.corr()

# Max correlation plot
#data.plot(kind='scatter',x='median_income',y='median_house_value', alpha = 0.2)

# 2D scatter plot
#data.plot(kind='scatter',x='longitude',y='latitude', alpha = 0.2)

data.plot(kind='scatter',x='longitude',y='latitude', alpha = 0.2,
          s=data['population']/100, label = 'population',
          c=data['median_house_value'], cmap=plt.cm.jet_r,
          colorbar=True)

# Subfig of all scatter plot
#pd.tools.plotting.scatter_matrix( data )

######################################
# Split data
###################################

train_set, test_set = sk.model_selection.train_test_split( data, test_size = 0.2, random_state = 42)

#train_set.hist( bins = 50 )

# Check size
#train_set.values.size
#data.values.size

######################################
#  New attributes
###################################

X = train_set.drop('median_house_value', axis=1)
y = train_set['median_house_value'].copy()

# Get rid of N/A values 
X.dropna(subset=['total_bedrooms'])

#  New attributes
X['rph'] = X['total_rooms']    / X['households']
X['bpr'] = X['total_bedrooms'] / X['total_rooms']
X['pph'] = X['population']     / X['households']

##########################################
## Encode text labels
#
#encoder = sk.preprocessing.LabelEncoder()
#
#oce_encoded = encoder.fit_transform( X['ocean_proximity'] )
#
## One-hot encoding
#
#hot = sk.preprocessing.OneHotEncoder()
#
#oce_hot = hot.fit_transform( oce_encoded.reshape(1, -1) )

######################################
# Encode binary label
######################################

encoder = sk.preprocessing.LabelBinarizer()

oce_encoded = encoder.fit_transform( X['ocean_proximity'] )