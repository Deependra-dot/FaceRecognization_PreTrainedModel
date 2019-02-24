# -*- coding: utf-8 -*-

import numpy as np
import model_inc
import add_new_face
import get_weight
import create_input_emb
from recog_face_in_cam import recognize_faces_in_cam

np.set_printoptions(threshold=np.nan)

newface='no'

newface=input('do you want to add new face to the DB? --> if yes type yes else type no\n')

while newface=='yes':
    add_new_face.add_new_face()
    newface=input('do you want to another face to the DB\n')
    
model= model_inc.dot_model()

weights, weights_dict = get_weight.get_weight()

# Set layer weights of the model
for name in weights:
  if model.get_layer(name) != None:
    model.get_layer(name).set_weights(weights_dict[name])
  elif model.get_layer(name) != None:
    model.get_layer(name).set_weights(weights_dict[name])
  
input_embeddings=create_input_emb.create_input_image_embeddings(model)


recognize_faces_in_cam(input_embeddings, model)

