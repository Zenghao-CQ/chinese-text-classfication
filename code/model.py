# -*- coding: utf-8 -*-  
from keras.utils import plot_model
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input,Dense,LSTM,Flatten,concatenate,Dropout,Masking,Reshape,Activation
from keras import regularizers
import os
from config import sent_num, \
                    batch_size, \
                    learning_rate,\
                    vec_dim, \
                    lstm_dim, \
                    class_num

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
    model = Model(input,output)
    omp = keras.optimizers.RMSprop(lr=learning_rate, epsilon=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer=omp, metrics=['accuracy']) 
    return model
    
if __name__ == "__main__":
    model = build_network()
    plot_model(model, to_file='model_bert_bilstm.png',show_shapes=True)
    model.summary()