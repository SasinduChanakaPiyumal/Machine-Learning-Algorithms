#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('iris.csv')


# In[4]:


data.sample(5)


# In[5]:


data.shape


# In[6]:


data.info()


# In[8]:


data.value_counts(['Species'])


# In[9]:


import seaborn as sns


# In[10]:


sns.pairplot(data,hue='Species')


# In[11]:


x = data.drop('Species',axis = 1)
x.sample(5)


# In[12]:


y = data['Species']
y.sample(5)


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[15]:


from sklearn.svm import SVC
model =SVC()   #can be change Search on Web
model.fit(x_train,y_train)


# In[16]:


model.kernel


# In[17]:


pred =model.predict(x_test)
pred


# In[18]:


y_test


# In[20]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(pred,y_test)


# In[21]:


confusion_matrix(pred,y_test)


# In[ ]:




