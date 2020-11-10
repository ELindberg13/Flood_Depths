import io
import os
import cv2
import numpy as np
from PIL import Image
from google.cloud import vision

from google.oauth2 import service_account

objects_to_crop_around = ['Car','Van','Truck','Boat','Toy vehicle']


# 
def crop_flooded_objects_boundary(google_api_key, file_list):

    credentials = service_account.Credentials. from_service_account_file(google_api_key)
    
    object_dict = {}
    cropped_images = []
    
    # creating a tenth and cycle counter to output progress of function
    tenth_counter = 0
    cycle_counter = 1
    
    #looping though each image in the list submitted to the function
    for file_item in file_list:
    
        #need to have google vision credentials saved to credentials
        client = vision.ImageAnnotatorClient(credentials=credentials)

        # path to the images that need to be cropped
        with open(file_item, 'rb') as image_file:
            content = image_file.read()
        image = vision.types.Image(content=content)
        
        #same path just using OpenCV to get image shape and will use to save the cropped images later
        im_cv2 = cv2.imread(file_item)
        height, width, color = im_cv2.shape

        #Using Google vision to actually find objects in the image
        objects = client.object_localization(image=image).localized_object_annotations
                
        tenth_counter += 1
        
        file_crops = []
        #looping through each of the objects Google vision found in the image
        for object_ in objects:
            # ignoring all objects that don't have to do with the cars in the image
            if object_.name in objects_to_crop_around:
                vertex_dict = {}

                #need to make sure the normalized vertex are multipled by the corresponding image distance so the vertex are in pixels counts
                for index,vertex in enumerate(object_.bounding_poly.normalized_vertices):
                    vertex_dict[f'vertex_{index}'] = [int(width*vertex.x),int(height*vertex.y)]
                object_dict[object_.name] = vertex_dict
            
                # Cropping the image around the vertices of the object
                
                # https://www.life2coding.com/cropping-polygon-or-non-rectangular-region-from-image-using-opencv-python/
                # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
                
                mask = np.zeros(im_cv2.shape[:2], np.uint8)
                points = np.array([object_dict[object_.name]['vertex_0'],
                                   object_dict[object_.name]['vertex_1'],
                                   object_dict[object_.name]['vertex_2'],
                                   object_dict[object_.name]['vertex_3']])
            
                #creating the bounding rectangle from the object vertices
                rect = cv2.boundingRect(points)
                x,y,w,h = rect
                
                # cropping the image using OpenCV and the dimentions of the bounding rectangle
                cropped = im_cv2[y:y+h, x:x+w].copy()

                file_crops.append(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))
                            
        cropped_images.append(file_crops)
        
    return cropped_images