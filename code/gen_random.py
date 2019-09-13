# -*- coding: utf-8 -*- 
import random
import io
import os
from config import data_path
fin_path = data_path+os.path.sep+"人教+北师大小学.txt"
fin = open(fin_path,'r',encoding='utf-8')
fout1 = open(data_path+os.path.sep+'train.txt','w',encoding='utf-8')
fout2 = open(data_path+os.path.sep+'valid.txt','w',encoding='utf-8')
train = [0,0,0,0,0,0]
valid = [0,0,0,0,0,0]

for line in fin:
    x = random.randint(0,9)
    if x<8:
        fout1.write(line)
        train[int(line[0])-1] += 1
    else:
        fout2.write(line)
        valid[int(line[0])-1] += 1

fin.close()
fout1.close()
fout2.close()

print('train:')
print(train)
print(sum(train))

print('valid:')
print(valid)
print(sum(valid))