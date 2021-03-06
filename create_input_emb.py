# -*- coding: utf-8 -*-
import glob
import os
import cv2
from image_to_emb import image_to_embedding


def create_input_image_embeddings(model):
    input_embeddings = {}

    for file in glob.glob("images/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(image_file, model)

    return input_embeddings
