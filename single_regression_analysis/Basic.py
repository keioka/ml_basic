#!/usr/bin/env python
# coding: utf-8

# a = 

# # numpy

# In[1]:


name = 'kei'


# In[3]:


import numpy as np


# In[15]:


x = np.array([1,2,3])


# In[17]:


y = np.array([2000, 3300, 4000])


# ### Centerling

# In[12]:


xc = x - x.mean()


# In[13]:


print(xc)


# ### Parameter

# In[14]:


xx = x * x


# In[18]:


yy = y * y


# In[20]:


xy = x * y


# In[21]:


xy


# # Pandas

# In[30]:


import pandas as pd


# In[33]:


df = pd.read_csv("sample.csv")


# In[36]:


df.head(3) # Show first 3 rows


# In[39]:


x = df['x']


# In[38]:


y = df['y']


# # Matplotilib

# In[41]:


import matplotlib.pyplot as plt


# In[46]:


plt.scatter(x, y) # Show scatter


# In[47]:


plt.show()


# # Data centraling using Pandas

# In[49]:


df.describe()


# In[50]:


df.mean()


# In[52]:


df_c = df - df.mean() # centralized data frame


# In[54]:


df_c.describe()


# In[55]:


df_c_x = df_c['x'] #centerlized x


# In[56]:


df_c_y = df_c['y'] #centerlized y


# In[57]:


plt.scatter(df_c_x, df_c_y)


# ### Get parameter a

# In[58]:


df_c_xx = df_c_x * df_c_x


# In[60]:


df_c_xy = df_c_x * df_c_y


# In[61]:


a = df_c_xy.sum() / df_c_xx.sum()


# In[62]:


a


# # Plot scatter 

# In[83]:


plt.scatter(df_c_x, df_c_y,   label='y', color='blue')
plt.plot(df_c_x, df_c_x*a,  label='y_hat', color='red')
plt.legend()
plt.show()


# # Prediction

# In[84]:


x_new = 40


# In[85]:


mean = df.mean()


# In[86]:


mean['x']


# In[87]:


xc = x_new -  mean['x']


# In[88]:


xc


# In[90]:


yc = a * xc


# In[91]:


y_hat = yc + mean['y']


# In[92]:


print(y_hat)


# In[101]:


def predict(x):
    mean = df.mean()
    xc = x - mean['x']
    ym = mean['y']
    y_hat = a * xc + ym
    return y_hat


# In[102]:


predict(30)


# In[105]:


predict(50)


# In[ ]:




