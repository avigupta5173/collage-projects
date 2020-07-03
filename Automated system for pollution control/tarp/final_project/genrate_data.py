import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import wave
import librosa

def remove_python_file(list_dir):
    clean_list=[]
    for file in list_dir:
        if(file[-3]!='.' and file[-2]!='p' and file[-1]!='y'):
            clean_list.append(file)
    return clean_list
def genrate_file_url(base_path,list_dir):
    genrate_url=[]
    for file in list_dir:
        genrate_url.append(os.path.join(base_path,file))
    return genrate_url

def genrate_new_dict(list_dir):
    new_folder = []
    for name in list_dir:
        val = 'img_' + name
        new_folder.append(val)
    return new_folder
def genrate_clean_filename(sound_files):
    name  = sound_files.split('.')
    return name[0]
#########################################
base_path = os.path.dirname(os.path.realpath(r'C:\Users\Lenovo\Desktop\tarp\final_project\*'))
list_dir = os.listdir(base_path)
#########################################
list_dir= remove_python_file(list_dir)
sound_directory_url = genrate_file_url(base_path,list_dir)
new_directory = genrate_new_dict(list_dir)
#########################################
for urls,folder in zip(sound_directory_url,new_directory):
    try:
        os.mkdir(folder)
        print("Directry Created: ",folder)
    except FileExistsError:
        print("File already Exists")
    new_folders_path = os.path.join(base_path,folder)
    print(new_folders_path)
    sound = os.listdir(urls)

    for sound_files in sound:
        print(folder)
        sound_file_path = os.path.join(urls,sound_files)
#        spf = wave.open(sound_file_path,'r')
        spf,_ = librosa.load(sound_file_path, sr=16000)
        signal = librosa.util.frame(spf,axis=-1)
#        signal = np.fromstring(signal, 'Int16')
#        fs = spf.getframerate()
        Time=np.linspace(0, len(signal)/_, num=len(signal))
        plt.figure(figsize=(4,2))
        plt.axis('off')
        plt.plot(Time,signal)
        plt.savefig(os.path.join(new_folders_path,str(genrate_clean_filename(sound_files))+'.png'))
        plt.show()
