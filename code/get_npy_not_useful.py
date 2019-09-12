import sys
import numpy as np
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    from keras_bert.datasets import get_pretrained, PretrainedList
    model_path = get_pretrained(PretrainedList.chinese_base)

vec_dim = 768
sent_len = 220
page_size = 184

paths = get_checkpoint_paths(model_path)
model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=sent_len)
print("***from zh:load bert model success***")
train_X = []
train_Y = []

token_dict = load_vocabulary(paths.vocab)
tokenizer = Tokenizer(token_dict)
fin = open('./train.txt','r',encoding='utf-8')
cnt=0
for line in fin:
	cnt += 1
	if(cnt%10==0):
		print("###")
		print(cnt)
		print("\n")
	train_Y.append(line[0])
	line = line[1:]
	temp = np.zeros((0,vec_dim))
	while line!=[]:
		text = line[0:512]
		line = line[512:]
		tokens = tokenizer.tokenize(text)
		indices, segments = tokenizer.encode(first=text, max_len=512)
		predicts = model.predict([np.array([indices]), np.array([segments])])[0]
		temp = np.concatenate((temp,predicts),axis=0)
	res = np.zeros((page_size-len(res),vec_dim))
	res = np.concatenate((res,temp),axis=0)
	train_X.append(res)

train_X = np.array(train_X)
train_Y = np.array(train_Y)
np.save("train_X.npy",train_X)
np.save("train_Y.npy",train_Y)