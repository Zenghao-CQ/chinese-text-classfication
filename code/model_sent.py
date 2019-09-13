# -*- coding: utf-8 -*-  
from keras.utils import plot_model
import keras
from keras.models import Model,Sequential
from keras.layers import Layer,Input,Dense,LSTM,Flatten,concatenate,Dropout,Bidirectional,Embedding,Reshape
from keras import regularizers
import keras.backend as K
import numpy as np
page_size = 1818#words of one squence
batch_size = 16
sent_num = 3
learning_rate = 0.001
vec_dim = 300#dim of wordvector
latent_dim = 64#dim of LSTM lay
latent_dim_1 = 256#dim of hidden lay
latent_dim_2 = 64#dim of hidden lay
class_num = 6

embedMatrix = np.zeros((187980 + 1,vec_dim))#words + stopword
femd = open("D:/NLP/文本分类/sgns.literature.bigram-char",'r',encoding='utf-8')
#Build embedding matrix
cnt = 0
femd.readline()
for line in femd:
	cnt += 1
	wlist = line.split(' ')
	key = wlist[0]#"key vector \n"
	wlist = wlist[1:-1]
	wlist = [float(x) for x in wlist]#str to float
	embedMatrix[cnt] = wlist

class SplitVector(Layer):
	def __init__(self, **kwargs):
		super(SplitVector, self).__init__(**kwargs)
	def call(self, inputs):
		# 按第二个维度对tensor进行切片，返回一个list
		in_dim = K.int_shape(inputs)[1]
		sent_len = in_dim//sent_num
		out = []
		for i in range(0,sent_num):
			out.append(inputs[:, i*sent_len:(i+1)*sent_len])
		return out
	def compute_output_shape(self, input_shape):
		# output_shape也要是对应的list
		in_dim = input_shape[1]
		sent_len = in_dim//sent_num
		return [(None,sent_len,)]*sent_num

def build_network(model_path = None):
	if model_path is not None:
		try:
			return keras.models.load_model(filepath = model_path)
		except OSError:
			print('Model path is not found')
	#build a new model
	model = Sequential()
	###Embedding
	in_ = Input(shape=(page_size,))	
	###split into vector
	splt = SplitVector()(in_)#return [sent,sent...]
	###embed
	embed = Embedding(input_dim = len(embedMatrix),
						output_dim = vec_dim,
                        weights=[embedMatrix],
						mask_zero = True,
                        trainable=False
						#input_length=page_size
						)
	splt = [embed(x) for x in splt]
	##sentence LSTM
	sent_lstm = Bidirectional(LSTM(latent_dim,
		return_sequences=False
		#kernel_regularizer=regularizers.l2(0.01)
		))
	sent_out = [sent_lstm(x) for x in splt]
	# ##page lstm
	res = Reshape((1,128))
	sent_out = [res(x) for x in sent_out]
	sent_out = concatenate(sent_out,axis = 1)
	page_lstm = Bidirectional(LSTM(latent_dim,
		return_sequences=False
		#kernel_regularizer=regularizers.l2(0.01)
		))
	page_out = page_lstm(sent_out)
	page_out = Dropout(0.3)(page_out)
	out_put = Dense(class_num, activation='softmax')(page_out)
	model = Model(in_, out_put)
	omp = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
	model.compile(loss='categorical_crossentropy', optimizer=omp, metrics=['accuracy'])
	return model
if __name__ == "__main__":
	model = build_network()
	plot_model(model, to_file='model.png',show_shapes=True)