# -*- coding: utf-8 -*-  
import numpy as np
from model import build_network
import keras
from keras.callbacks import TensorBoard
import os
train_X = np.load("../data/train_X_f.npy")
train_Y = np.load("../data/train_Y_f.npy")
valid_X = np.load("../data/valid_X_f.npy")
valid_Y = np.load("../data/valid_Y_f.npy")
#x = keras.preprocessing.sequence.pad_sequences(x,maxlen=6,dtype= x.dtype, padding='pre',value=0.)
epochnum = 70
page_size = 1818#words of one squence
batch_size = 15
learning_rate = 0.0005
vec_dim = 300#dim of wordvector
class_num = 6

modelpath = '../model/model_d50.h5'
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

print("****data load success...")
model = build_network()
#model = build_network("../model/epoch.h5")
#model.optimizer.lr = learning_rate
his = model.fit( train_X, train_Y,
    batch_size=batch_size,
    epochs=epochnum, 
    verbose=1,
    shuffle=True,
    initial_epoch=0,
    callbacks = [back],#,tbCallBack],
    validation_data = (valid_X,valid_Y))