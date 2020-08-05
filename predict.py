import sys
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
from model_data import ResNet50


model = ResNet50(include_top=False, load_weight=True, weights='models/run224_0.01_weights.854-1.66_final.hdf5',
                input_shape=(224,224,3))


# %% Run MAIN

'''
Za poednostavno koristenje od Spyder 4.0, kodot e strukturiran da koristi use_arg, kako vlez.
Kako parametri prima jpg, png, i mp4 fajlovi pri shto:
    jpg i png gi kreira vo plots delot
    mp4 kako video stream koj se zapira so 'q' bukvata
'''
img_name = "evning_test.jpg" 
vid_name = "project_video.mp4"

use_arg = img_name

def parseImage(img): 
    '''
    Ja prasira individualnata slika i vrsi predviduvanje i 
    crtanjeto na kutii vrz nea
    '''
    print(img.shape)
    dims = (img.shape[1], img.shape[0])
    
    img_float = cv2.resize(img, (224,224)).astype(np.float32)
    img_float -= 128
    
    img_in = np.expand_dims(img_float, axis=0)
    
    pred = model.predict(img_in)
    
    bboxes = utils.get_boxes(pred[0], dims=dims, cutoff=0.2)
    bboxes = utils.nonmax_suppression(bboxes, iou_cutoff = 0.05)
    draw = utils.draw_boxes(img, bboxes, color=(0, 0, 255), thick=3, draw_dot=True, radius=3)
    draw = draw.astype(np.uint8)
    return draw
#sys.argv[1]



print(use_arg)
if use_arg.lower().endswith(('.png', '.jpg')):
    
    img = cv2.imread(img_name)
    draw = parseImage(img)
    
    plt.imshow(draw[...,::-1])
    plt.show()
elif use_arg.lower().endswith(('.mp4')):
    cap = cv2.VideoCapture(use_arg) 
    count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            parsed = parseImage(frame)
            cv2.imshow('window-name',parsed)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows() 

