#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[45]:


from sklearn.datasets import fetch_california_housing


# In[97]:


data = fetch_california_housing()
a = pd.read_csv(r"C:\Users\lenovo\Desktop\Ml Project-2\housing.csv")
b = data.target


# In[98]:


a


# In[100]:


a.drop(['longitude','latitude','median_house_value'], axis=1,inplace=True)


# In[101]:


a.info()


# In[102]:


a


# In[103]:


a.hist(bins=50,figsize=(20,15))


# In[104]:


a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2,random_state=100)


# In[105]:


a.shape


# In[106]:


b.shape


# In[107]:


def poly(degree):
    
    poly_features = PolynomialFeatures(degree=degree)
    
    a_train_poly = poly_features.fit_transform(a_train)
    a_test_poly = poly_features.fit_transform(a_test)
    
    regression=LinearRegression()
    regression.fit(a_train_poly,b_train)
    
    b_train_predict_poly=regression.predict(a_train_poly)
    b_test_predict_poly=regression.predict(a_test_poly)
    
    rmse_train=(np.sqrt(mean_squared_error(b_train,b_train_predict_poly)))
    rmse_test=(np.sqrt(mean_squared_error(b_test,b_test_predict_poly)))
    
    r2_train=r2_score(b_train,b_train_predict_poly)
    r2_test=r2_score(b_test,b_test_predict_poly)
    
    
    print("Train Mean Squared Error is :", rmse_train)
    print("Test Mean Squared  Error is :", rmse_test)
    print("Train R2 Score is  :", r2_train)
    print("Test R2 Score is  :", r2_test)


# In[108]:


poly(2)


# In[109]:


X1 = pd.DataFrame(np.c_[a['median_income'],a['population']],columns=['median_income','Population'])
Y1 = data.target


# In[110]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)


# In[111]:


def poly_new(degree):
    
    poly_features = PolynomialFeatures(degree=degree)
    
    X1_train_poly = poly_features.fit_transform(X1_train)
    X1_test_poly = poly_features.fit_transform(X1_test)
    
    regression=LinearRegression()
    regression.fit(X1_train_poly,Y1_train)
    
    Y1_train_predict_poly=regression.predict(X1_train_poly)
    Y1_test_predict_poly=regression.predict(X1_test_poly)
    
    rmse_train=(np.sqrt(mean_squared_error(Y1_train,Y1_train_predict_poly)))
    rmse_test=(np.sqrt(mean_squared_error(Y1_test,Y1_test_predict_poly)))
    
    r2_train=r2_score(Y1_train,Y1_train_predict_poly)
    r2_test=r2_score(Y1_test,Y1_test_predict_poly)
    
    
    print("Train Mean Squared Error is :", rmse_train)
    print("Test Mean Squared  Error is :", rmse_test)
    print("Train R2 Score is  :", r2_train)
    print("Test R2 Score is  :", r2_test)


# In[112]:


poly_new(2)


# In[ ]:




