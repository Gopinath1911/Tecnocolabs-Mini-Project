#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

df=pd.read_csv("train.csv")


# In[2]:


df.head()


# In[9]:


def gen_freq(text):
    #Will store the list of words
    word_list = []

    #Loop over all the tweets and extract words into word_list
    for tw_words in text.split():
        word_list.extend(tw_words)

    #Create word frequencies using word_list
    word_freq = pd.Series(word_list).value_counts()

    print(word_freq[:20])
    
    return word_freq

gen_freq(df.comment_text.str)


# In[ ]:




