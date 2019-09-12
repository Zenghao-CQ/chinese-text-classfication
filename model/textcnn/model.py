# -*- coding: utf-8 -*-  
from keras.utils import plot_model
import keras
from keras.models import Model,Sequential
from keras.layers import *
from keras import regularizers
from keras import backend as K
import numpy as np
page_size = 1818#words of one squence
batch_size = 15
learning_rate = 0.0005
vec_dim = 300#dim of wordvector
kl_size = [2,3,4]##kernal size
fl_size = 50##filter szie
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

def build_network(model_path = None):
	if model_path is not None:
		try:
			return keras.models.load_model(filepath = model_path)
		except OSError:
			print('Model path is not found')
	#build a new model
	###Embedding
	in_put = Input(shape=(page_size,))	
	###embed
	embed = Embedding(input_dim = len(embedMatrix),
						output_dim = vec_dim,
                        weights=[embedMatrix],
						mask_zero = False,
                        trainable=False,
						input_length=page_size
						)
	in_ = embed(in_put)
	conv_out = []
	##convolution
	for kl in kl_size:
		conv_t = Conv1D(filters = fl_size,kernel_size = kl,activation = "tanh")(in_)
		pool_t = MaxPooling1D(K.int_shape(conv_t)[1])(conv_t)
		pool_t = Flatten()(pool_t)
		conv_out.append(pool_t)
	conv_out = concatenate(conv_out,axis = 1)
	conv_out = Dropout(0.5)(conv_out)
	##dense
	out_put = Dense(class_num, activation='softmax',kernel_regularizer=regularizers.l1(0.01))(conv_out)
	model = Model(in_put, out_put)
	omp = keras.optimizers.RMSprop(lr=learning_rate, epsilon=1e-06)
	model.compile(loss='categorical_crossentropy', optimizer=omp, metrics=['accuracy'])
	plot_model(model, to_file='model.png',show_shapes=True)
	return model
build_network()
