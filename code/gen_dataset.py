# -*- coding: utf-8 -*- 
import random
import io
import os
dirin = "d:/NLP/文本分类/人教"
fout1 = open('d:/NLP/文本分类/data/train.txt','w',encoding='utf-8')
fout2 = open('d:/NLP/文本分类/data/valid.txt','w',encoding='utf-8')
stopwoeds = [' ','/','a','b','c','d','e','f','g','h','i','j','k',
'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
files = os.listdir(dirin)
maxlen = 0
for i in range(0,len(files)):
    path = dirin+'/'+files[i]
    fin = open(path,'r', encoding='utf-8')
    line = fin.readline()
    content = files[i][0]
    for word in line:
        if word not in stopwoeds:
            content += word
    maxlen = max(maxlen,len(content))
    content += '\n'
    x = random.randint(0,9)
    if(x<8):
        fout1.write(content)
    else:
        fout2.write(content)
    fin.close()
print("maxlen:")
print(maxlen)