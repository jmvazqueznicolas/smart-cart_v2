#!/usr/bin/env python3

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

import numpy as np

# Se extraen las carcateriticas de la imagen
class FeatureExtractor:
    def __init__(self, modelo):
        if modelo == 1:
            base_model = MobileNet(include_top=False, weights='imagenet')
            self.model = Sequential([base_model, GlobalAveragePooling2D()])
            print('Extracción con MobileNet')
        elif modelo == 2:
            base_model = ResNet152(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            print('Extracción con ResNet152')
        elif modelo == 3:
            base_model = VGG16(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
            print('Extracción con VGG16')

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(1024, )
        """
        #img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        #img = img.convert('RGB')  # Make sure img is color
        #x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(img, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 1024) -> (1024, )
        #print("feature",feature)
        #self.model.summary()
        return feature / np.linalg.norm(feature)  # Normalize