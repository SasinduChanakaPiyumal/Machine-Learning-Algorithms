#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[8]:


data = pd.read_csv("sample.csv")


# In[9]:


data.head()


# In[10]:


data.tail()


# In[15]:


import matplotlib.pyplot as plt

plt.scatter(data.age, data.job)
plt.show()


# In[16]:


x = data[['age']]


# In[17]:


x


# In[18]:


y = data['job']


# In[19]:


y


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[22]:


x_test


# In[23]:


x_train


# In[24]:


model = LogisticRegression()


# In[25]:


model.fit(x_train,y_train)


# In[26]:


model.predict(x_test)


# In[27]:


model.score(x_test,y_test)


# In[28]:


age = np.array([[20],[25],[30],[35]])
model.predict(age)


# In[ ]:




