#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[2]:


df = pd.read_csv('news.csv')
df.head()


# In[3]:


labels = df.label
labels.head()


# In[4]:


x_test,x_train,y_test,y_train = train_test_split(df['text'],labels,test_size=0.2,random_state=7)


# In[5]:


#intializing TfidfVectorizer
Tfidf_Vectorizer = TfidfVectorizer(stop_words ='english',max_df=0.7)

#fit and transform train set
tfidf_train = Tfidf_Vectorizer.fit_transform(x_train)
tfidf_test = Tfidf_Vectorizer.transform(x_test)


# In[6]:


#intializing PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# In[7]:


#prediction on test set and calculating accuracy
y_pred = pac.predict(tfidf_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'accuracy_score:{accuracy*100}%')


# In[11]:


confusion_matrix(y_test,y_pred)


# In[ ]:




