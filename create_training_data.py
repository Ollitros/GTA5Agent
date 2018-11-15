# create_training_data.py
from util.grabscreen import grab_screen
from util.getkeys import key_check
import numpy as np
import cv2
import time
import os


def keys_to_output(keys):

    """
    Convert keys to a ...multi-hot... array

    [A,W,D,S] boolean values.

    """

    output = [0, 0, 0, 0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[3] = 1
    return output


file_name = 'data/training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0, 40, 800, 640))
            last_time = time.time()
            screen = cv2.resize(screen, (299, 299))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])
            
            if len(training_data) % 50 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
        if 'P' in keys:
            break

main()
