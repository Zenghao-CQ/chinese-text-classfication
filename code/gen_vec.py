# -*- coding: utf-8 -*- 
import jieba
import io
import numpy as np
train_X = []
embed = {}
fin = open('D:/NLP/文本分类/data/valid.txt','r',encoding='utf-8')
femd = open("D:/NLP/文本分类/sgns.literature.bigram-char",'r',encoding='utf-8')

#Build dictionary
femd.readline()
cnt=0
vec_dim = 300
page_size = 1818
for line in femd:
    cnt+=1
    wlist = line.split(' ')
    key = wlist[0]#"key vector \n"
    jieba.add_word(key)
    embed[key] = cnt

#Pre embedding
cnt = 0
train_X = []
train_Y = []
for line in fin:
    lable = line[0]
    y = [0]*6
    y[int(lable)-1] = 1
    train_Y.append(y) 
    wlst = jieba.lcut(line[1:-1],cut_all=False)#line[0] is class of text,line[-1] is\n
    wlst = [(embed[x] if x in embed else 0) for x in wlst]
    train_X.append([0]*(page_size-len(wlst)) + wlst)

train_X = np.array(train_X,dtype=np.int32)
train_Y = np.array(train_Y)
fin.close()
femd.close()
print(train_X.shape)
print(train_Y.shape)
np.save("valid_X_f.npy",train_X)
np.save("valid_Y_f.npy",train_Y)