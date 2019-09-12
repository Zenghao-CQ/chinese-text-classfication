# -*- coding: utf-8 -*- 
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
import numpy as np


vec_dim = 768
sent_len = 220#220
sent_num = 288#284

file_path = '../data/人教+北师大中小学.txt'
model_path =  "/home/alex/bert_cls/chinese_L-12_H-768_A-12"
fin = open(file_path,'r',encoding='utf-8')
paths = get_checkpoint_paths(model_path)
model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=sent_len)

token_dict = load_vocabulary(paths.vocab)
tokenizer = Tokenizer(token_dict)

def insidef(text):
    indices, segments = tokenizer.encode(first=text, max_len=sent_len)
    predicts = model.predict([np.array([indices]), np.array([segments])])[0]
    return predicts[0]

all_ = []
for cnt,line in enumerate(fin):
    print('*****line %d start*****' % cnt)
    line = line[:-1].split('##')# lable & \n
    lable = line[0]
    line = line[1:]
    vecs = [insidef(x) for x in line]
    vecs = np.array(vecs)
    all_.append(vecs)
    print(vecs.shape)
all_ = np.array(all_)
print(all_.shape)
np.save("bert.npy",all_,allow_pickle=True)