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
# Split data
###################################

train_set, test_set = sk.model_selection.train_test_split( data, test_size = 0.2, random_state = 42)


######################################
#  Split input/ouputs
###################################

X_raw = train_set.drop('median_house_value', axis=1)
y_raw = train_set['median_house_value'].copy()


######################################
#  Split numerical/text data
###################################

class NumSelector( sk.base.BaseEstimator , sk.base.TransformerMixin ):
    def __init__(self):
        """ Select numerical column and return numpy array """
        pass
        
    def fit(self, X , y=None ):
        """ Fit transform params"""
        return self
    
    def transform(self, X, y=None ):
        """ Transform input data """
        return X.drop('ocean_proximity', axis=1 ).values
        
        
class TexSelector( sk.base.BaseEstimator , sk.base.TransformerMixin ):
    def __init__(self):
        """ Select the text column and return numpy array """
        pass
        
    def fit(self, X , y=None ):
        """ Fit transform params"""
        return self
    
    def transform(self, X, y=None ):
        """ Transform input data """
        return X['ocean_proximity'].values
        

##########################
# Fill-in NA values
##########################

#filler = sk.preprocessing.Imputer()

######################################
# Encode binary label
######################################

#encoder = sk.preprocessing.LabelBinarizer()


######################################
# New attributes with custom class
######################################

class AttirbutesAdder( sk.base.BaseEstimator , sk.base.TransformerMixin ):
    def __init__(self, add_bpr = True ):
        self.add_bpr = add_bpr
        
    def fit(self, X , y=None ):
        """ Fit transform params"""
        return self
    
    def transform(self, X, y=None ):
        """ Transform input data """
        rph = X[:,3] / X[:,6]
        pph = X[:,5] / X[:,6]
        
        if self.add_bpr:
            
            bpr = X[:,4] / X[:,3]
            return np.c_[X,rph,pph,bpr]
        
        else:
            return np.c_[X,rph,pph]
        

        

######################################
# Combine all features transforms
######################################

X_num = X_raw.drop('ocean_proximity', axis=1 ).values
X_tex = X_raw['ocean_proximity']
   
######################################
# Combine all features transforms
######################################


            
num_pipeline = sk.pipeline.Pipeline([
        ('num_sel', NumSelector() ),                         # Select num values
        ('filler', sk.preprocessing.Imputer() ),             # Fille NaN with median
        ('new_att', AttirbutesAdder() ),                     # Augmented with ratios
        ('std_scaler', sk.preprocessing.StandardScaler(), )  # Normalized data
        ])
    
tex_pipeline = sk.pipeline.Pipeline([
        ('tex_sel', TexSelector() ),                         # Select tex values
        ('encoder', sk.preprocessing.LabelBinarizer() ),      # From text to binary values
        ])    

full_pipeline = sk.pipeline.FeatureUnion( transformer_list =[
        ('num_pipeline', num_pipeline ),                         
        ('tex_pipeline', tex_pipeline ),
        ])
    
######################################
# Preprocessing
######################################
    
X_num   = num_pipeline.fit_transform( X_raw )
    
tex_sel = TexSelector()
X_tex   = tex_sel.fit_transform( X_raw )

encoder = sk.preprocessing.LabelBinarizer()
X_bin   = encoder.fit_transform( X_tex )

X = np.c_[ X_num , X_bin ]

y = y_raw.values

######################################
# Regression
######################################

from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor
from sklearn.ensemble     import RandomForestRegressor

#reg = sk.linear_model.LinearRegression()
#reg = sk.tree.DecisionTreeRegressor()
reg = sk.ensemble.RandomForestRegressor()

reg.fit( X , y )


######################################
# Predictions
######################################

x_samples = X[1:10,:]
y_true    = y[1:10]
y_hat     = reg.predict( x_samples )


######################################
# Evaluation
######################################

rms = np.sqrt( sk.metrics.mean_squared_error( y_true, y_hat ) )

#print('True: ', y_true,'\nPredi:',y_hat,'\nRMS:',rms)

# Cross evaluation

from sklearn.model_selection import cross_val_score

RMSs = np.sqrt( -cross_val_score( reg , X , y , scoring='neg_mean_squared_error', cv=10) )
print('Average Error:',RMSs.mean())


