# -*- coding: utf-8 -*-
from image_to_emb import image_to_embedding
import numpy as np
from scipy import spatial



def recognize_face(face_image, input_embeddings, model):

    embedding = image_to_embedding(face_image, model)
    #print(embedding)
    
    minimum_distance = 200
    max_similarity = 2
    name = None
    
    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():
        
       
        euclidean_distance = np.linalg.norm(embedding-input_embedding)
        result = 1 - spatial.distance.cosine(embedding, input_embedding)

        

        print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))
        print('Cosime Similarity from %s is %s' %(input_name, result))
        

        
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
            max_similarity=result
    return str(name), str(max_similarity)
    #if minimum_distance < 0.68:         
        #return str(name), str(max_similarity)
        #return str(name)
    #if minimum_distance < 0.68:
        #return str(name)
    #else:
        #return None