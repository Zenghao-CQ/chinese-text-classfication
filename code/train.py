# -*- coding: utf-8 -*- 
import os 
import numpy as np
from model import build_network
import keras
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from config import epoch_num, \
                    sent_num, \
                    batch_size, \
                    learning_rate,\
                    vec_dim, \
                    class_num, \
                    data_path

train_X = np.load(os.path.join(data_path,"train_X.npy"),allow_pickle = True)
train_Y = np.load(os.path.join(data_path,"train_Y.npy"),allow_pickle = True)
valid_X = np.load(os.path.join(data_path,"valid_X.npy"),allow_pickle = True)
valid_Y = np.load(os.path.join(data_path,"valid_Y.npy"),allow_pickle = True)

train_X = pad_sequences(train_X, maxlen=sent_num, dtype= 'float32', padding='pre',value=0.)
train_Y = to_categorical(train_Y)
valid_X = pad_sequences(valid_X, maxlen=sent_num, dtype= 'float32', padding='pre',value=0.)
valid_Y = to_categorical(valid_Y)
print("****data load success...")
modelpath = '../model/model_bert_lstm.h5'
#modelpath = '../model/model{epoch:03d}.h5'
boardpath = './logs'

back = keras.callbacks.ModelCheckpoint(modelpath,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='max', period = 1)

tbCallBack = TensorBoard(log_dir=boardpath,  # log 目录
                histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                batch_size=batch_size,     # 用多大量的数据计算直方图
                write_graph=True,  # 是否存储网络结构图
                write_grads=True, # 是否可视化梯度直方图
                write_images=True,# 是否可视化参数
                embeddings_freq=0, 
                embeddings_layer_names=None, 
                embeddings_metadata=None)

model = build_network()
#model = build_network("../model/epoch.h5")
#model.optimizer.lr = learning_rate
his = model.fit( train_X, train_Y,
    batch_size=batch_size,
    epochs=epoch_num, 
    verbose=1,
    shuffle=True,
    initial_epoch=0,
    callbacks = [tbCallBack],#back
    validation_data = (valid_X,valid_Y))