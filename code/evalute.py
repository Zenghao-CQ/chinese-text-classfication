import numpy as np
from model import build_network
import keras
import os
valid_X = np.load("../data/valid_X_new.npy")
valid_Y = np.load("../data/valid_Y.npy")

model_path = "../model/model004.h5"
model = keras.models.load_model(filepath = model_path)
loss,acc = model.evaluate(x=valid_X, y=valid_Y )
print('loss ',loss,' acc ',acc)
