#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('creditcard.csv')


# In[2]:


df.head()


# In[2]:


df.shape


# In[3]:


df.isnull().count()


# In[4]:


df.isnull().sum()


# In[6]:


df['Class'].value_counts()


# In[7]:


x = df.drop("Class",axis =1)

y = df.Class


# ### cross validation like KFold and Hyperparameter tunning in unbalanced datasets

# In[8]:


from sklearn.ensemble import RandomForestClassifier


# In[9]:


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score , confusion_matrix,classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

import numpy as np


# In[18]:


log_class = LogisticRegression()

# hyperparameter tunning 

grid = {"C": 10.0 **np.arange(-2,3),'penalty':['l1','l2']}


cv = KFold(n_splits = 5,random_state=None,shuffle = False)


# In[19]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test= train_test_split(x,y,train_size=0.7)


# In[20]:


clf = GridSearchCV(log_class, grid,cv=cv,n_jobs=-1,scoring='f1_macro')

clf.fit(x_train,y_train)


# In[21]:


y_pred=clf.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


# In[25]:


y_train.value_counts()


# In[26]:


class_weight = dict({0:1,1:100})


# In[27]:


from  sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(class_weight=class_weight)

classifier.fit(x_train,y_train)


# In[28]:


y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


# ### Under Sampling 

# In[31]:


from collections import Counter
from imblearn.under_sampling import NearMiss

ns = NearMiss(0.8)

x_train_ns,y_train_ns = ns.fit_sample(x_train,y_train)


print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes before fit {}".format(Counter(y_train_ns)))


# In[32]:


classifier.fit(x_train_ns,y_train_ns)


# In[33]:


y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


# ##### undersampling  --- reduce the points of the maximum labels 
# 
# 

# ### Over Sampling 

# In[34]:


from imblearn.over_sampling import RandomOverSampler


# In[39]:


os = RandomOverSampler(0.75)

x_train_ns , y_train_ns = os.fit_sample(x_train,y_train)


# In[40]:


print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes before fit {}".format(Counter(y_train_ns)))


# In[41]:


from  sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(class_weight=class_weight)

classifier.fit(x_train_ns,y_train_ns)


# In[42]:


y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


# ###  SmoteTomek --- Smothe technique

# In[43]:


from imblearn.combine import SMOTETomek


# In[45]:


os = RandomOverSampler(0.75)

x_train_ns , y_train_ns = os.fit_sample(x_train,y_train)

print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes before fit {}".format(Counter(y_train_ns)))


# In[46]:


from  sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(class_weight=class_weight)

classifier.fit(x_train_ns,y_train_ns)


# In[47]:


y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


# ### Ensemble technique 

# In[48]:


from imblearn.ensemble import EasyEnsembleClassifier


# In[49]:


easy = EasyEnsembleClassifier()
easy.fit(x_train,y_train)


# In[50]:


y_pred=easy.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### under Sampling 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




