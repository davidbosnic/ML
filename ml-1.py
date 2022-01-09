#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import matplotlib.pyplot as mtp
import numpy as nm


# In[65]:


data_set= pd.read_csv('Desktop/ObesityDataSet_raw_and_data_sinthetic.csv')


# In[66]:


data_set


# In[78]:


x= data_set.iloc[:,:-1].values


# In[79]:


x


# In[80]:


y= data_set.iloc[:,16].values  
y


# In[81]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])
x[:, 4]= label_encoder_x.fit_transform(x[:, 4]) 
x[:, 5]= label_encoder_x.fit_transform(x[:, 5]) 
x[:, 8]= label_encoder_x.fit_transform(x[:, 8]) 
x[:, 9]= label_encoder_x.fit_transform(x[:, 9]) 
x[:, 11]= label_encoder_x.fit_transform(x[:, 11])  
x[:, 14]= label_encoder_x.fit_transform(x[:, 14])  
x[:, 15]= label_encoder_x.fit_transform(x[:, 15])  
x[0]


# In[71]:


labelencoder_y= LabelEncoder()  
y= labelencoder_y.fit_transform(y) 
y


# In[89]:


onehot_encoder= OneHotEncoder()    

d= onehot_encoder.fit_transform(x[:, [14, 15]]).toarray()

d[0]


# In[88]:


x[:, [14, 15]]


# In[ ]:




