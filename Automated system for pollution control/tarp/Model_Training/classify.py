import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import model_from_json
from keras.optimizers import SGD
import cv2
import os
import wave 

def cvt_to_batch(ar):
    batch=[]
    bat = len(ar)/5
    last=0
    while(last<len(ar)):
        batch.append(ar[int(last):int(last+bat)])
        last+=bat
    return batch
def only_image(lst):
    l=[]
    for img_list in lst:
        val = img_list.split('.')
        try:
            if(val[1]=='png'):
                l.append(img_list)
        except:
            pass
    return l
    
#########################################3

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

loaded_model.compile(SGD(lr=0.1,),loss='categorical_crossentropy',metrics=['accuracy'])
#--------------------------------------
spf = wave.open('yam.wav','r')
#--------------------------------------
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
Time=np.linspace(0, len(signal)/fs, num=len(signal))

batch=[]
bat = len(signal)/5
last=0
while(last<len(signal)):
    batch.append(signal[int(last):int(last+bat)])
    last+=bat
batch_signal = np.array(batch)

batch=[]
bat = len(signal)/5
last=0
while(last<len(Time)):
    batch.append(Time[int(last):int(last+bat)])
    last+=bat
batch_time = np.array(batch)

for i in range(5):
    plt.figure(figsize=(4,2))
    plt.axis('off')
    plt.plot(batch_time[i],batch_signal[i])
    plt.savefig("test"+str(i)+".png")
    plt.show()
    

file_cont = os.path.dirname(os.path.realpath('classify'))
list_dir = only_image(os.listdir(file_cont))
result=[]
for name in list_dir:
    img = cv2.imread(name)
    img = cv2.resize(img,(64,64))
    image = np.array([img])
    result.append(loaded_model.predict_classes(image))
    
for the_file in list_dir:
    file_path = os.path.join(file_cont, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
        
for i in range(len(result)):
    if result[i][0]==0:
        print("crack shaft")
    elif result[i][0]==1:
        print("fan")
    else:
        print("piston")