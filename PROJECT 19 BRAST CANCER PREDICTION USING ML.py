#!/usr/bin/env python
# coding: utf-8

# IMPORTING THE DEPENDENCES

# In[7]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# DATA COLLECTION AND PROCESSING

# In[14]:


#loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[15]:


print(breast_cancer_dataset)


# In[16]:


# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# In[17]:


# print the first 5 rows of the dataframe
data_frame.head()


# In[18]:


#adding the target column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# In[19]:


#print last 5 rows of the dataframe
data_frame.tail()


# In[22]:


# number of rows and coulmns in the dataset
data_frame.shape


# In[23]:


# getting some information about the data
data_frame.info()


# In[24]:


# checking for missing values 
data_frame.isnull().sum()


# In[25]:


#statistical measures about the data
data_frame.describe()


# In[26]:


#checking the distribution of target variable
data_frame['label'].value_counts()


# 1 = BENIGN
# 0 = MALIGNANT

# In[28]:


data_frame.groupby('label').mean()


# SEPARATING THE FEATURES AND TARGET

# In[29]:


X = data_frame.drop(columns= 'label', axis = 1)
Y = data_frame['label']


# In[30]:


print(X)


# In[31]:


print(Y)


# SPLITTING THE DATA INTO TRAINING AND TEST DATA

# In[32]:


X_train, X_test, Y_train,Y_test = train_test_split(X,Y,random_state= 2, test_size = 0.2)


# In[33]:


print(X.shape,X_train.shape, X_test.shape)


# MODEL TRAINING 

# LOGISTIC REGRESSION

# In[34]:


model = LogisticRegression()


# In[35]:


# training the logistic Regression model using the training data

model.fit(X_train, Y_train)


# MODEL EVALUATION

# ACCURACY SCORE

# In[38]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)


# In[39]:


print('Accuracy on training data = ', training_data_accuracy)


# In[41]:


#accuracy on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test,X_test_prediction)


# In[42]:


print('Accuracy on test data = ', test_data_accuracy)


#  BUILDING A PREDICTIVE SYSTEM

# In[47]:


input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshapre the numpy array as we are predicting for one datapoint

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)

print(prediction)

if (prediction[0] == 0):
    print('The breast cancer is Malignant')
else:
    print('The Breast Cancer is Benign')


# In[ ]:




