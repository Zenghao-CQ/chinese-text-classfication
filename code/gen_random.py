# -*- coding: utf-8 -*- 
import random
import io
import os
fin_path = "./data/人教+北师大小学.txt"
fin = open(fin_path,'r',encoding='utf-8')
fout1 = open('./data/train.txt','w',encoding='utf-8')
fout2 = open('./data/valid.txt','w',encoding='utf-8')
train = [0,0,0,0,0,0]
valid = [0,0,0,0,0,0]

for line in fin:
    x = random.randint(0,9)
    if x<8:
        fout1.write(line+'\n')
        train[int(line[0])-1] += 1
    else:
        fout2.write(line+'\n')
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