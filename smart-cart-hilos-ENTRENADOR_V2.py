import threading
import cv2
import time
import numpy as np
import os
import re

camaras = ['/dev/video0','/dev/video1','/dev/video2']

#Estas lineas son para configurar las camaras Logitech
for camara in camaras:
    os.system(f"v4l2-ctl -d {camara} --list-ctrls")
    os.system(f"v4l2-ctl -d {camara} --set-ctrl=exposure_auto=1")
    os.system(f"v4l2-ctl -d {camara} --set-ctrl=exposure_absolute=80")
    os.system(f"v4l2-ctl -d {camara} --list-ctrls")

# Manipulacion de hilos
width = 480 
height = 320 

frame1 = np.array([])
frame2 = np.array([])
frame3 = np.array([])

flag1 = False
flag2 = False
flag3 = False

detener_hilos = False
flag_tree_images = False

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) 
    while True:
        ret, frame2 = cap.read()
        flag2 = True
        if detener_hilos == True:
            cap.release()
            break 

def cam_3(w, h):
    global frame3
    global flag3
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    while True:
        ret, frame3 = cap.read()
        flag3 = True
        if detener_hilos == True:
            cap.release()
            break   

cam1 = threading.Thread(target=cam_1, args=(width,height,))
cam2 = threading.Thread(target=cam_2, args=(width,height,))
cam3 = threading.Thread(target=cam_3, args=(width,height,))

cam1.start()
cam2.start()
cam3.start()

# Ciclo principal
terminar = False
cwd = os.getcwd()
pahimg = cwd + '/static/img'
filename_list = []

while terminar == False:
    # Se pregunta si las camaras estan listas para ser leidas
    if ((flag1 == True) and (flag2 == True) and (flag3 == True)):
        """
        frames = [frame1.copy(), frame2.copy(), frame3.copy()]
        for cam, frame in enumerate(frames, 1):
            if cam==1:
                cv2.imshow('Camara 1', frame)
            elif cam==2:
                cv2.imshow('Camara 2', frame)
            elif cam==3:
                cv2.imshow('Camara 3', frame)
        """

        start_key = input("Press 'a' to introduce new code or 'q' to quit ")
        #if cv2.waitKey(50) & 0xFF == ord('a'):
        if start_key == 'a':
            codigo = str(input("Muestre el codigo de barras: "))
            #codigo = re.sub("[^0-9]","",codigo)
            print(f"El codigo es:\n{codigo}")
            time.sleep(1)

            i=0
            more_pics = input("Take pictures? (y/n): ")
            frames = [frame1, frame2, frame3]
            while (more_pics == 'y'):
                for cam, frame in enumerate(frames, 1):
                    image_np = np.array(frame)

                    filename = codigo+ "_" + str(i) + ".jpg"
                    filename_list.append(filename)
                    frameCopy = frame.copy()
                    cv2.imwrite(os.path.join(pahimg,filename),frameCopy)
                    i+=1
                    
                    if cam==1:
                        cv2.imshow('frame1', frame)
                    elif cam==2:
                        cv2.imshow('frame2', frame)
                    elif cam==3:
                        cv2.imshow('frame3', frame)
                    
                more_pics = input("Take pictures? (y/n)")
                time.sleep(1) 

        elif start_key == 'q':
            detener_hilos = True
            terminar = True
            cv2.destroyAllWindows()
            break

modelo = input("""Select the model: \n 1.- Mobilenet \n 2.- Resnet152 \n 3.- VGG16 \n """)

comando = f"./offline_features.py {modelo}"

os.system(comando)