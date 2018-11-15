import numpy as np
import keras
from keras import models, layers, utils, backend, optimizers, losses, datasets, applications
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications import InceptionResNetV2, Xception
from keras.models import load_model, model_from_json


def save(model):
    # Save Sequential in two files
    model_json = model.to_json()
    with open('data/models/model.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights('data/models/model.h5')
    print("Saved model to disk")


# Load model from json and weights
def load():
    # Load model
    path = 'data/models/model.json'
    json_file = open(path, 'r')
    # json_file = open(self.path_to_model_json, 'r')
    print('in preidcuit')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('data/models/model.h5')
    print("Loaded model from disk")

    return model


def create_model():
    # Create model
    # model = models.Sequential()
    #
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(299, 299, 3), name='block1_conv1'))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='block1_conv2'))
    # model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
    # model.add(Dropout(0.25, name='block1_dropout1'))
    #
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='block2_conv1'))
    # model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool1'))
    # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='block2_conv2'))
    # model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool2'))
    # model.add(Dropout(0.25, name='block2_dropout1'))
    #
    # model.add(Flatten(name='block3_flatten'))
    # model.add(Dense(512, activation='relu', name='block3_dense1'))
    # model.add(Dropout(0.5, name='block3_dropout1'))
    # model.add(Dense(3, activation='softmax', name='block3_dense2'))

    # model = InceptionResNetV2(weights=None, classes=4)
    model = Xception(weights=None, input_shape=(299, 299, 3), classes=4)

    return model


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    WIDTH = 299
    HEIGHT = 299

    train_data = np.load('data/train/balanced_dataset/balanced_dataset.npy')

    train = train_data[:-100]
    test = train_data[-100:]
    X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
    Y = [i[1] for i in train]
    Y = np.array((Y))

    test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
    test_y = [i[1] for i in test]
    print(type(Y))

    # Train model
    model = create_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
                  metrics=['accuracy'])
    model.fit(X, Y, batch_size=12, epochs=50)

    save(model)


if __name__ == "__main__":
    main()