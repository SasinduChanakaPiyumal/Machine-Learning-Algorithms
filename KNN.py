#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('iris.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data['Species'].value_counts()


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


x = data.iloc[:,0:4]
x.head()


# In[10]:


y = data.iloc[:,-1]
y.head()


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
x[0:5]


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[13]:


x_train.shape


# In[14]:


x_test.shape


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)


# In[16]:


pred = model.predict(x_test)
pred[0:5]


# In[17]:


y_test[0:5]


# In[18]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,pred)


# In[19]:


accuracy


# In[20]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
cm


# In[21]:


result = pd.DataFrame(data=[y_test.values,pred], index = ['y_test','pred'])
result.transpose()


# In[22]:


correct_sum =[]
for i in range(1,20):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    correct = np.sum(pred == y_test)
    correct_sum.append(correct)


# In[23]:


correct_sum


# In[26]:


result = pd.DataFrame(data=correct_sum)
result.index = result.index+1
result.T


# In[28]:


model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train,y_train)
pred = model.predict(x_test)
accuracy_score(y_test,pred)


# In[29]:


confusion_matrix(y_test,pred)


# In[ ]:




