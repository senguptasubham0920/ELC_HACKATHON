# ELC_HACKATHON
The script consists for 3 modules . All of them are developed in python.

1. Relate - This includes detecting the objects from the images shown to the consumers who are normal and have sense of smell.  
We have used Single Shot Detector framework combined with the MobileNet architecture as our deep learning-based object detector. 
MobileNet is a lightweight and fast object detector model that was developed by Google. It was trained on the ImageNet dataset. 
For identifying the objects(labels) we have used the coco_names label dataset and the model used is ssd_mobilenet_v3_large_coco. 

2. Generate - Objective is to generate the top objects (keywords) from the relatable images.we create a dataset which contains the following information : 
Image names,
Objects detected in the image,
Whether the image was relatable to the scent of the perfume. 

Then we create a competency score for each object in the following manner 
The relative frequency of the objects in all relatable images, 
The relative co-occurence of each word along with other words in the relatable images,  
Competency Score = Weighted mean of the relative frequency and relative co-occurence, 
    
3. Generate images -Generative AI based models to create images from top relatable keywords. We use  DALL.E2 from OpenAI API in python .
