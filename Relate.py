#!/usr/bin/env python
# coding: utf-8

# Relate -
# Participants in the survey, with normal sense of smell, identify the images they relate to with the perfume. .
# 
# We have used Single Shot Detector framework combined with the MobileNet architecture as our deep learning-based object detector. MobileNet is a lightweight and fast object detector model that was developed by Google. It was trained on the ImageNet dataset
# 
# We have used the following steps for training -
# 
# 1. Mention the location of the image which is to be detected.
# 2. Inlude the weights and model file for training
# 3. Include the appropriate class names or labels (here we have used coco_names label dataset)
# 4. Adjust the minimum confidence reagrding the model to detect the object
# 

# In[ ]:


#import libraries 
import cv2
import numpy as np
import pandas as pd


# In[ ]:


#insert the path of the image to be used for detection 
def Relate_func(img_path,threshold):
    image = cv2.imread(img_path)
    
    #resize the image
    image = cv2.resize(image, (640, 480))
    
    #store the shape and size
    h = image.shape[0]
    w = image.shape[1]

    # path to the weights and model files
    weights = "C:/Users/SUBHAM SENGUPTA/Downloads/ELC_HACKATHON/config/frozen_inference_graph.pb"
    model = "C:/Users/SUBHAM SENGUPTA/Downloads/ELC_HACKATHON/config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    # load the MobileNet SSD model trained  on the COCO dataset
    net = cv2.dnn.readNetFromTensorflow(weights, model)

    # load the class labels the model was trained on
    class_names = []
    with open("C:/Users/SUBHAM SENGUPTA/Downloads/ELC_HACKATHON/config/coco_names.txt", "r") as f:
        class_names = f.read().strip().split("\n")

    # create a blob from the image
    blob = cv2.dnn.blobFromImage(
    image, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
    # pass the blog through our network and get the output predictions
    net.setInput(blob)
    output = net.forward()  # shape: (1, 1, 100, 7)

    class_ids = []
    probabilities = []
    boxes = []
    # loop over the number of detected objects
    for detection in output[0, 0, :, :]:  # output[0, 0, :, :] has a shape of: (100, 7)
        # the confidence of the model regarding the detected object
        probability = detection[2]

        # if the confidence of the model is lower than 50%,
        # we do nothing (continue looping)
        if probability < threshold:
            continue

        # perform element-wise multiplication to get
        # the (x, y) coordinates of the bounding box
        box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
        box = tuple(box)
        # draw the bounding box of the object
        cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), thickness=2)

        # extract the ID of the detected object to get its name
        class_id = int(detection[1])
        class_ids.append(class_id)
        probabilities.append(float(probability))

        # draw the name of the predicted object along with the probability
        label = f"{class_names[class_id - 1].upper()} {probability * 100:.2f}%"
        #cv2.putText(image, label, (box[0], box[1] + 15),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return class_names,class_ids


