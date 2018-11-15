# test_model.py

import numpy as np
from utils.grabscreen import grab_screen
import cv2
import time
from utils.directkeys import PressKey, ReleaseKey, W, A, S, D
from keras.applications import InceptionResNetV2
from utils.getkeys import key_check
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.xception import Xception
from keras.models import load_model
from keras import optimizers
from train_model import save, load

WIDTH = 299
HEIGHT = 299
LR = 1e-3
EPOCHS = 10

t_time = 0.09


def straight():
    ##    if random.randrange(4) == 2:
    ##        ReleaseKey(W)
    ##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    PressKey(W)
    PressKey(A)
    # ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    # ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)


def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    # ReleaseKey(W)
    # ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)


def stop():
    ##    if random.randrange(4) == 2:
    ##        ReleaseKey(W)
    ##    else:
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)


def main():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Train model
    loaded_model = load()
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            # 800x600 windowed mode
            # screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0, 40, 800, 640))
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            screen = cv2.resize(screen, (299, 299))

            prediction = loaded_model.predict([screen.reshape(1, 299, 299, 3)])
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[0][1] > fwd_thresh:
                straight()
            elif prediction[0][0] > turn_thresh:
                left()
            elif prediction[0][2] > turn_thresh:
                right()
            elif prediction[0][3] > turn_thresh:
                stop()
            else:
                straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(1)


main()