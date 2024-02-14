#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('Kyphosis.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[6]:


data.info()


# In[8]:


x = data.drop('Kyphosis',axis=1)


# In[9]:


x.head()


# In[10]:


y = data['Kyphosis']
y.head()


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[12]:


x_train.shape


# In[13]:


x_test.shape


# ## Desition Tree

# In[16]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# In[19]:


pred = model.predict(x_test)


# In[20]:


y_test


# In[21]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# In[22]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)


# In[ ]:




