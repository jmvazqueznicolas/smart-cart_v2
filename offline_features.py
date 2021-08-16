#!/usr/bin/env python3

from feature_extractor import FeatureExtractor
import numpy as np
import os, sys
import cv2

"""
This version extracts the feature vector of the given image files if the filenames 
are given as argument and if not, it searches in the whole folder for the image 
files in order to create the feature vectors
"""

def main():
    modelo = int(sys.argv[1])
    print("Running offline mobilenet with the given image files")    
    fe = FeatureExtractor(modelo)
    cwd = os.getcwd()
    path = cwd + "/static/img"
        
    print("Creating feature vectors for all the images in the folder")
    files = os.listdir( path )

    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for img_path in files:
        flag_ext = False
        for ext in extensions:
            if (img_path.endswith(ext)):
                flag_ext = True
                break
        if (not flag_ext):
            ("Removing filename ", img_path)
            files.remove(img_path)

    for img_path in files:
        curr_img_path = os.path.join(path, img_path)
        frame = cv2.imread(str(curr_img_path))
        frame = cv2.resize(frame, (224,224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np = np.array(frame)
        img_path = img_path.rsplit('.',1)
        feature = fe.extract(image_np)
        feature_filename = img_path[0] + ".npy"
        if modelo == 1:
            feature_path = os.path.join(cwd, "static/feature/mobilenet", feature_filename)
        elif modelo == 2:
            feature_path = os.path.join(cwd, "static/feature/resnet152", feature_filename)
        elif modelo == 3:
            feature_path = os.path.join(cwd, "static/feature/VGG16", feature_filename)
        np.save(feature_path, feature)

if __name__ == '__main__':
    main()
