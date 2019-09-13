import numpy as np
from model import build_network
import keras
import os
from config import data_path,model_path
valid_X = np.load(os.path.join(data_path,"valid_X_new.npy"))
valid_Y = np.load(os.path.join(data_path,"valid_Y.npy"))

model_name = "model004.h5"
path = os.path.join(model_path,model_name)
model = keras.models.load_model(filepath = path)
loss,acc = model.evaluate(x=valid_X, y=valid_Y )
print('loss ',loss,' acc ',acc)
