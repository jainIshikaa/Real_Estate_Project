#!/usr/bin/env python
# coding: utf-8

# # REAL ESTATE PRICE PREDICTOR

# In[1]:


import sklearn


# In[2]:


import pandas as pd


# In[3]:


housing=pd.read_csv("data.csv")


# In[4]:


#housing.head()


# In[5]:


#housing.info()


# In[6]:


#housing["CHAS"].value_counts()


# In[7]:


#housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


housing.hist(bins=50,figsize=(20,15))


# ## TRAIN-TEST Splitting

# In[11]:


# import numpy as np
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     test_set_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]


# In[12]:


#train_set,test_set=split_train_test(housing,0.2)


# In[13]:


#print(f"Rows in train set:{len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["CHAS"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[16]:


strat_test_set["CHAS"].value_counts()


# In[17]:


housing=strat_train_set.copy()


# ## Looking for Correlations

# In[18]:


#corr_matrix=housing.corr()
#corr_matrix["MEDV"].sort_values(ascending=False)


# In[19]:


#corr_matrix["MEDV"].sort_values(ascending=False)


# In[20]:


#from pandas.plotting import scatter_matrix
#attributes=["MEDV","RM","ZN","LSTAT"]
#scatter_matrix(housing[attributes],figsize=(12,8))


# In[21]:


#housing.plot(kind="scatter",x="RM",y="MEDV",alpha=1)


# ## Trying out Attribute Combinations

# In[22]:


#housing["TAXRM"]=housing["TAX"]/housing["RM"]


# In[23]:


#housing["TAXRM"]


# In[24]:


housing.head()


# In[25]:


#corr_matrix=housing.corr()
#corr_matrix["MEDV"].sort_values(ascending=False)


# In[26]:


#housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=1)


# In[27]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[28]:


median=housing["RM"].median()


# In[29]:


median


# In[30]:


housing["RM"].fillna(median)


# In[31]:


housing.describe()# before started missing attributes(imputer)


# In[32]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[33]:


imputer.statistics_


# In[34]:


X=imputer.transform(housing)


# In[35]:


housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[36]:


housing_tr.describe()


# ## Scikit-learn Design
# 
# Primarily three types of objects
# 1.Estimators-It estimates some parameters based on the dataset.Eg.imputer.It has a fit method and transform method.Fit method-Fits the dataset and calculates internal parameters.
# 2.Transformers-transform method takes input and return output based on the learnings from fit().It also has a convenience method called fit_transform() which fits and then transforms.
# 3.Predictors-Linear REgression method is an example .fit()and predict() are two common functions.It also gives score() function which will evaluate the prediction.

# ## Feature Scaling
# 
# Primarily,two types of feature scaling methods:
# 
# 
# 
# 1.Min-Max Scaling(Normalisaton)
#     (value-min)/(max-min)
#     
#     
# 2.Standarisation
#     (value-mean)/std
#     SK learn provides a class called StandardScaler for this

# ## Creating Pipeline

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("std_scaler",StandardScaler()),])


# In[38]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)
# can also usw (housing) because imputer is mentioned in pipeline


# In[39]:


housing_num_tr.shape


# ## Selecting a desired model for Real Estate

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[41]:


some_data=housing.iloc[:5]


# In[42]:


some_labels=housing_labels.iloc[:5]


# In[43]:


prepared_data=my_pipeline.transform(some_data)


# In[44]:


model.predict(prepared_data)


# In[45]:


list(some_labels)


# ## Evaluating the model
# 

# In[46]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[ ]:


rmse #Not good because labels are also low,trying another model(decisiontreeregressor)


# ## Using better evaluation technique-Cross Validation
#                     (previous model(DTR) overfiited)

# k-fold method
# divide dataset into equal parts(let 10) and compute mean std of each and then takeout conclusion about mean_squared_error

# In[ ]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)# cross_val_score not works for minimum(as cost) but for utility(should be max),so compute negative and to take sqrrt -ve is converted to +ve


# In[ ]:


rmse_scores


# In[ ]:


def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Standard Deviation: ",scores.std())


# In[ ]:


print_scores(rmse_scores)


# ## Saving the model

# In[ ]:


from joblib import dump,load
dump(model,"Dragon.joblib")


# ## Testing the model on Test Data

# In[ ]:


X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(Y_test))


# In[ ]:


final_rmse


# In[ ]:


prepared_data[0]


# ## Using the model

# In[ ]:


from joblib import dump,load
import numpy as np
model=load("Dragon.joblib")
features=np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.9449947 , -1.31238772,  2.3211401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




