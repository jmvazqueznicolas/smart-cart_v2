#!/usr/bin/env python3
import numpy as np
import cv2
import threading
from feature_extractor import FeatureExtractor
import os, sys, time

"""
camaras = ['/dev/video0','/dev/video2','/dev/video4']

#Estas lineas son para configurar las camaras Logitech
for camara in camaras:
    os.system(f"v4l2-ctl -d {camara} --list-ctrls")
    os.system(f"v4l2-ctl -d {camara} --set-ctrl=exposure_auto=1")
    os.system(f"v4l2-ctl -d {camara} --set-ctrl=exposure_absolute=250")
    os.system(f"v4l2-ctl -d {camara} --list-ctrls")
"""

def nothing(x):
    pass

# Manipulación de hilos
width = 480 #480 #640 #1280
height = 320 #320 #480 #720

frame1 = np.array([])
frame2 = np.array([])
frame3 = np.array([])

flag1 = False
flag2 = False
flag3 = False

detener_hilos = False

def cam_1(w, h):
    global frame1
    global flag1
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) 
    while True:
        ret, frame1 = cap.read()
        flag1 = True
        if detener_hilos == True:
            cap.release()
            break

def cam_2(w, h):
    global frame2
    global flag2
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) # 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) # 1080
    while True:
        ret, frame2 = cap.read()
        flag2 = True
        if detener_hilos == True:
            cap.release()
            break     

def cam_3(w, h):
    global frame3
    global flag3
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) # 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) # 1080
    while True:
        ret, frame3 = cap.read()
        flag3 = True
        if detener_hilos == True:
            cap.release()
            break   


def get_product_name(texto):
    texto = texto.rsplit('/',3)
    texto = texto[3]
    texto = texto.rsplit('_',1)
    return str(texto[0])


cam1 = threading.Thread(target=cam_1, args=(width,height,))
cam2 = threading.Thread(target=cam_2, args=(width,height,))
cam3 = threading.Thread(target=cam_3, args=(width,height,))
cam1.start()
cam2.start()
cam3.start()

# Selección del modelo desde la terminal
modelo = int(input("""Select the model: \n 1.- Mobilenet \n 2.- Resnet152 \n 3.- VGG16 \n """))

cwd = os.getcwd()
if modelo == 1:
    print('MobileNet')
    features_root_path = cwd + "/static/feature/mobilenet/"
elif modelo == 2:
    print('ResNet152')
    features_root_path = cwd + "/static/feature/resnet152/"
elif modelo == 3:
    print('VGG16')
    features_root_path = cwd + "/static/feature/VGG16/"


# Read image features
fe = FeatureExtractor(modelo)
features = []
img_paths = []


features_files = os.listdir(features_root_path)
extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
for filename in features_files:
    if (not filename.endswith(".npy")):
        features_files.remove(filename)
    else:
        feature_path = features_root_path + filename
        features.append(np.load(feature_path))

        fn = filename.rsplit('.',1)
        curr_image_path = cwd + "/static/img/" + fn[0]
        for ext in extensions:
            if (os.path.isfile(curr_image_path + ext)):
                img_paths.append(curr_image_path + ext)
                break
            else:
                print("No image found for this feature")
features = np.array(features)
#print("features",features)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 50)
org2 = (20, 90)
fontScale = 1
color = (255, 0, 0)
thickness = 2

#Ciclo principal
#num_frames = 60
#curr_frame = 0
terminar = False

cv2.namedWindow('Camara 1')
cv2.namedWindow('Camara 2')
cv2.namedWindow('Camara 3')
cv2.namedWindow('Resultado')
cv2.createTrackbar('Dist', 'Camara 1', 0, 100, nothing)

while terminar == False:
    # Se pregunta si las cámaras estan listas para ser leidas
    if ((flag1 == True) and (flag2 == True) and (flag3 == True)):
        frames = [frame1, frame2, frame3]
        best_scores = []
        scores = []
        names = []
        # Se procesan los frames    
        for cam, frame in enumerate(frames, 1):
           
            # Run search
            frame_cp = frame.copy()   # Se crea una copia
            frame = cv2.resize(frame, (224,224))    # Needs an RGB image of 224x224 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np = np.array(frame)
            query = fe.extract(image_np)

            #t0 = time.time()
            dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            #t1 = time.time()
            #tiempo = t1 - t0
            #print("total time ", tiempo)
            #print("time per feature", tiempo / len(features))

            ids = np.argsort(dists)[:1]  # Top result
            best_scores.append((dists[ids[0]], img_paths[ids[0]]))

            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
            #cv2.imshow('Camara 1', image_np)
            if cam == 1:
                cv2.imshow('Camara 1', frame_cp)
            elif cam == 2:
                cv2.imshow('Camara 2', frame_cp)
            elif cam == 3:
                cv2.imshow('Camara 3', frame_cp)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                detener_hilos = True
                terminar = True
                cv2.destroyAllWindows()
                break

        # Se crean listas para el manejo de los scores y nombres
        for products in best_scores:
            scores.append(products[0])
            names.append(get_product_name(products[1]))
            
        # Umbral de detección
        umbral = 75
        
        # Si el producto se encuentra por debajo del umbral, entonces se tiene certeza
        if(100*min(scores) < umbral):
            index_min = min(range(len(scores)), key=scores.__getitem__)
            result = cv2.putText(frames[0], names[index_min], org, font, fontScale, color, thickness, cv2.LINE_AA)
            result = cv2.putText(frames[0], str(int(100*min(scores))), org2, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            result = cv2.putText(frames[0], "Umbral no alcanzado", org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Se muestra la cámara frontal para visualización del resultado
        cv2.imshow('Resultado', result)
