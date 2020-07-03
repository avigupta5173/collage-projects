import os
from keras.models import load_model
# test_model.py
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey,space
from getkeys import key_check
import random
os.chdir('C:\\Users\\avigu\\OneDrive\\Desktop\\image project')
space = [1,0]
def space_press():
 PressKey(space)
 ReleaseKey(space)
model=load_model('last.hdf5')
last_time = time.time()
for i in list(range(4))[::-1]:
 print(i+1)
 time.sleep(1)
paused = False
while(True):

 if not paused:
 # 800x600 windowed mode
 screen = grab_screen(region=(650,155,1250,257))
 print('loop took {} seconds'.format(time.time()-last_time))
 last_time = time.time()
 screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
 screen = cv2.bitwise_not(screen)
 screen = cv2.resize(screen, (60,60))
 mean = np.mean(screen,axis=(0,1))
 std = np.std(screen,axis=(0,1))
 screen = (screen-mean)/(std+1e-7)

 prediction = model.predict(screen.reshape(1,60,60,1))[0]
 print(prediction)
 if np.argmax(prediction) == np.argmax(space):
 space_press()
 keys = key_check()
 # p pauses game and can get annoying.
 if 'T' in keys:
 if paused:
 paused = False
 time.sleep(1)
 else:
 paused = True
 ReleaseKey(space)
 time.sleep(1)
