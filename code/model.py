# -*- coding: utf-8 -*-  
from keras.utils import plot_model
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input,Dense,LSTM,Flatten,concatenate,Dropout,Masking,Reshape,Activation
from keras import regularizers
import os
sent_num = 176#nums of one squence 169
              #Max senteces length:153,useless
batch_size = 15
learning_rate = 0.0005
vec_dim = 728#dim of wordvector
lstm_dim = 100
class_num = 6

def build_network(model_path=None):
    if model_path is not None:
        try:
            return keras.models.load_model(filepath = model_path)
        except OSError:
            print('Model path is not found')
    input = Input(shape=(sent_num,vec_dim),name="input")
    masked = Masking(mask_value=0.,input_shape = (sent_num, vec_dim), 
                    name="mask_zero")(input)
    f_lstm = LSTM(units = lstm_dim, input_shape = (sent_num, vec_dim),
                    dropout = 0.2, name = 'LSTM_F')(masked)#foward
    b_lstm = LSTM(units = lstm_dim, input_shape = (sent_num, vec_dim),
                    dropout = 0.2, name = 'LSTM_B')(masked)#backward
    x = concatenate([f_lstm,b_lstm],axis = 1,name = 'concat')
    x = Activation(activation='tanh',input_shape = (lstm_dim*2, vec_dim),
                    name = 'activate')(x)
    x = Dropout(0.5,input_shape = (lstm_dim*2, vec_dim),name= 'dropout')(x)
    output = Dense(6,activation='softmax',input_shape = (lstm_dim*2, vec_dim),
                    name = 'softmax')(x)
    return Model(input,output)
    

if __name__ == "__main__":
    model = build_network()
    plot_model(model, to_file='model_bert_bilstm.png',show_shapes=True)
    model.summary()