#!/usr/bin/env python
# coding: utf-8

# Generative AI based models to generate images from top relatable keywords
# 
# We create imgaes with the top relatable keywords. We use DALL.E2 from OpenAI.

# In[14]:


#import libraries 
import openai 
import os
import requests





# In[28]:


def gen_img(API_KEY,PROMPT,count,name):
    
    openai.api_key = API_KEY
    

    response = openai.Image.create(
    prompt=PROMPT,
    n=count,
    size="1024x1024",)
    
    image_url = response['data']
    image_url = [image["url"] for image in image_url]
    
    for url,name in zip(image_url,name):
        image = requests.get(url)
        with open("{}.png".format(name), "wb") as f:
            f.write(image.content)

