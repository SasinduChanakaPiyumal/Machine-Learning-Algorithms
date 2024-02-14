#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
data = np.array([[26,50000],[29,70000],[34,55000],[31,41000]])


# # Normalization

# In[5]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data =scaler.fit_transform(data)
scaled_data


# # Standardization

# In[8]:


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data


# In[ ]:




