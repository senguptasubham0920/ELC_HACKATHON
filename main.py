#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import numpy as np
import pandas as pd
import openai

import Relate
import Generate
import Generate_img


# Relate - Detect the objects from image relatable to the scent of the perfume

# In[3]:


#mention the image path for object detection
img_path = 'C:/Users/SUBHAM SENGUPTA/Downloads/ELC_HACKATHON/data/input_imgs/img4.jpg'
#provide the threshold for model to detect the object
threshold = 0.5

class_names,class_ids = Relate.Relate_func(img_path,threshold)

for i in range(len(class_ids)):
    print(class_names[class_ids[i]-1])


# Generate - Top objects from the relatable images 

# In[4]:


#loading the dataset
data = pd.read_csv("C:/Users/SUBHAM SENGUPTA/Downloads//ELC_HACKATHON/data/generate.csv")
print("Structure of generated dataset from all images")
print(data.head())

#selecting only those images which are relatable to the perfume 
data = data[data['Relateable'] == 1]

#calling function to generate top keywords 
Generate.Generate_func(data)


# Generate Image - Use Generative AI to create images using the top relatable objects

# In[5]:


#provide the API_KEY
API_KEY = "OPENAI API-KEY"
#provide the number of images to be created
count = 2

#provide the relateable objects
PROMPT = "person dog tree bagpack ballon"

#names of images to be stored
name = ['C:/Users/SUBHAM SENGUPTA/Downloads/ELC_HACKATHON/data/created_images/cr_1',
       'C:/Users/SUBHAM SENGUPTA/Downloads/ELC_HACKATHON/data/created_images/cr_2']

Generate_img.gen_img(API_KEY,PROMPT,count,name)

