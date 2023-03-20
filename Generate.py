#!/usr/bin/env python
# coding: utf-8

# Generate_Images -
# 
# This module generates the top relatable keyword from 
# 
# we create a dataset which conatains the following information :
# 
# 1. Images
# 2. Objects deteted
# 3. Whether the image was relatable to the scent of the perfume.
# 
# We consider the images that were relatable to a particular perfume. We measure the top relatable keywords from the images.
# 
# The competency score for each word is calculated in the following manner - 
# 1. The relative frequency of the objects in all relatable images
# 2. The relative co-occurence of each word along with other words in the relatable images 
# 3. Weighted mean of the relative frequency and relative co-occurence

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import re
import nltk
import string 
import os
from collections import Counter


# In[2]:


#import nltk
#nltk.download('punkt')


# In[3]:


#relate function which choose the top relatable keywords from the images

def preprocessing(corpus):
    # initialize
    clean_text = []

    for row in corpus:
        # tokenize
        tokens = nltk.tokenize.word_tokenize(row)
        # lowercase
        tokens = [token.lower() for token in tokens]
        # isword
        tokens = [token for token in tokens if token.isalpha()]
        clean_sentence = ''
        clean_sentence = ' '.join(token for token in tokens)
        clean_text.append(clean_sentence)
        
    return clean_text


def Generate_func(full_df):
    # filtering only those images whcih the participants could relate  
    all_text = preprocessing(full_df.Objects)
    
    # sklearn countvectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    # Convert a collection of text documents to a matrix of token counts
    cv = CountVectorizer(ngram_range=(1,1), stop_words = 'english')
    
    
    X = cv.fit_transform(all_text)
    Xc = (X.T * X)# matrix manipulation
    names = cv.get_feature_names_out() # This are the entity names (i.e. keywords)
    df = pd.DataFrame(data = Xc.toarray(), columns = names,index = names)
    df['cooccurence_sum'] = df.sum(axis=1) - np.diag(df)
    print("\nThe co-occurence matrix\n")
    print(df)
    
    final = pd.concat([pd.DataFrame(np.diag(df)/np.sum(np.diag(df)),df.index),df['cooccurence_sum']/np.sum(df.cooccurence_sum)],axis = 1)
    final.rename(columns = {0:'Frequency'},inplace = True)
    
    freq_wt = 0.5 
    con_wt = 0.5
    final['competency_score'] = (final.Frequency*freq_wt + final.cooccurence_sum*con_wt)*100
    print("\nThe final competency score for each objects\n")
    print(final)
    
    final = final.nlargest(5,'competency_score')
    listToStr = ' '.join([str(elem) for elem in final.index])
    
    print('\nTop related objects from the related images')
    
    return  listToStr



