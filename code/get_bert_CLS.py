# -*- coding: utf-8 -*- 
from keras_bert import extract_embeddings
import numpy as np

file_path = './data/人教+北师大中小学.txt'
model_path =  "/home/alex/bert_cls/chinese_L-12_H-768_A-12"
fin = open(file_path,'r',encoding='utf-8')

vec_dim = 768
sent_len = 220#220
sent_num = 288#284

#all_ = []
for cnt,line in enumerate(fin):
    print('*****line %d start*****' % cnt)
    line = line[:-1].split('##')# lable & \n
    lable = line[0]
    line = line[1:]
    vecs = extract_embeddings(model_path, line)
    vecs = [x[0] for x in vecs]
    vecs = np.array(vecs)
    print(vecs.shape)

    #res = np.zeros((sent_num-len(vecs),vec_dim))
    #vecs = np.concatenate((res,vecs),axis=0)
    #all_.append(vecs)

    np.save("train_line_"+str(cnt)+".npy",vecs)
#train_X = np.array(all_)