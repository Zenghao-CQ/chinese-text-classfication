# -*- coding: utf-8 -*- 
import io
import os
import re

end_of_sent = '##'
pattern = "[。！？”……]"
stop_word = ["“","”"]

max_sent_num = -1
max_sent_len = -1

def read_sigle_text(flie_path):
    global max_sent_len
    global max_sent_num
    file_class = os.path.basename(flie_path)[0]
    fin = open(flie_path,'r',encoding='utf-8')
    total = []
    for line in fin:
        line = line[:-1]
        temp = filter(lambda x:x not in stop_word,line)#return yeild in py3!attention,use list(xx)
        out = "".join(temp)
        temp = re.split(pattern,out)#there may be some ''
        temp = filter(lambda x:x != "",temp)
        total += temp
    sent_num = len(total)
    sent_len = max([len(x) for x in total])
    max_sent_num = max(max_sent_num,sent_num)
    max_sent_len = max(max_sent_len,sent_len)
    total = file_class+end_of_sent+end_of_sent.join(total)
    return total

def read_dir(dir_path,fout):
    cnt = [0,0,0,0,0,0,0,0,0]
    files = os.listdir(dir_path)
    for file in files:
        flie_path = os.path.join(dir_path,file)
        content = read_sigle_text(flie_path)
        cnt[int(content[0])-1]+=1
        fout.write(content+'\n')
    return cnt

        
write_path = "./data/人教+北师大中小学.txt"
fout = open(write_path,'w',encoding='utf-8')
dir_path = './文本/人教+北师大中小学'
cnt = read_dir(dir_path,fout)
print("Max senteces numers:",end='')
print(max_sent_num)
print("Max senteces length:",end='')
print(max_sent_len)
print("size of each grades:")
print(cnt)