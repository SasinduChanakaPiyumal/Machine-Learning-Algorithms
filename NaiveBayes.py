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


# In[5]:


data.info()


# In[6]:


import seaborn as sns
sns.pairplot(data,hue ="Kyphosis")


# In[7]:


sns.countplot(x = "Kyphosis",data=data)


# ## Data Pre-Processing

# In[8]:


x = data.drop('Kyphosis',axis=1)


# In[9]:


x.head()


# In[10]:


y = data['Kyphosis']
y.head()


# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[22]:


x_train.shape


# In[23]:


x_test.shape


# ## Naive Bayes

# ### Gaussian

# In[24]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train, y_train)


# In[25]:


pred = NB.predict(x_test)
pred


# In[26]:


y_test


# In[27]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# In[28]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)


# In[29]:


data['Kyphosis'].value_counts()


# ### Multinomial

# In[43]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train,y_train)


# In[44]:


pred = model.predict(x_test)
pred


# In[45]:


y_test


# In[46]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# In[47]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)


# ### Bernoulli

# In[49]:


from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(x_train,y_train)


# In[50]:


pred = model.predict(x_test)
pred


# In[51]:


y_test


# In[52]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# In[53]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)


# In[ ]:




