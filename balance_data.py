# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('data/train/training_dataset.npy')
shape = train_data.shape
print(shape)
train_data = np.reshape(train_data, [shape[0]*shape[1], shape[2]])
print(train_data.shape)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []
stops = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0, 0]:
        lefts.append([img, choice])
    elif choice == [0, 1, 0, 0]:
        forwards.append([img, choice])
    elif choice == [0, 0, 1, 0]:
        rights.append([img, choice])
    elif choice == [0, 0, 0, 1]:
        stops.append([img, choice])
    else:
        print('no matches')

forwards = forwards[:len(lefts)][:len(stops)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
stops = stops[:len(forwards)]

final_data = forwards + lefts + rights + stops
shuffle(final_data)
np.save('data/train/balanced_dataset/balanced_dataset.npy', final_data)




