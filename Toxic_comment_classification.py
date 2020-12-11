#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[20]:


import gc
import time
import warnings

#stats

from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


train.head()


# In[22]:


test.head()


# In[24]:


train.isnull().sum()


# In[25]:


test.isnull().sum()


# In[35]:


train.info()


# In[37]:


test.info()


# In[55]:


train['no_tag']=train.iloc[:,2:].sum(axis=1)
train.no_tag.value_counts()


# In[58]:


train.no_tag.value(42)


# In[42]:


no_tag=train.iloc[:,2:].sum(axis=1)
Train['clean']=1 
if (no_tag==0): 
else: 
(0)


# In[33]:


x=train.iloc[:,2:].sum()

plt.figure(figsize=(8,4))
ax=sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Tag classification", fontsize=15)
plt.xlabel("Divisons", fontsize=14)
plt.ylabel("No._of_comments", fontsize=14)

rects = ax.patches
labels = x.values

for rect, label in zip(rects, labels):
    height=rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha="center", va='bottom')

plt.show()


# In[ ]:




