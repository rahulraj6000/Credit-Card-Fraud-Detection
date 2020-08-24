#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd

import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn

plt.style.use('ggplot')






df = pd.read_csv('creditcard.csv')


# In[ ]:





# In[20]:


df.head()


# In[21]:


df.describe()


# In[22]:


df.info()


# In[23]:


df.shape


# In[ ]:





# In[24]:


df.hist(figsize = (20, 20))
plt.show()


# In[25]:


#visualizations of time and amount
plt.figure(figsize=(10,8))
plt.title('Distribution of Time Feature')
sns.distplot(df.Time)


# In[11]:


plt.figure(figsize=(10,8))
plt.title('Distribution of Monetary Value Feature')
sns.distplot(df.Amount)


# In[26]:


# correlation matrix
corr_matrix = df.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corr_matrix, vmax = .8, square = True)
plt.show()


# In[27]:


counts = df["Class"].value_counts()
normal = counts[0]
fraudulent = counts[1]
perc_normal = (normal/(normal+fraudulent))*100
perc_fraudulent = (fraudulent/(normal+fraudulent))*100
print('There were {} non-fraudulent transactions ({:.3f}%) and {} fraudulent transactions ({:.3f}%).'.format(normal, perc_normal, fraudulent, perc_fraudulent))


# In[28]:


plt.figure(figsize=(8,6))
sns.barplot(x=counts.index, y=counts)
plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions')
plt.ylabel('Count')
plt.xlabel('Class (0:Non-Fraudulent, 1:Fraudulent)')


# In[30]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=False)
plt.show();


# In[31]:


df.isnull().count()


# In[32]:


df.isnull().sum()


# In[ ]:





# In[ ]:




