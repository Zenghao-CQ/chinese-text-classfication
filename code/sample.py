import numpy as np
import random
X = np.load('./data/bert.npy',allow_pickle=True).tolist()
Y = np.load('./data/lables.npy',allow_pickle=True).tolist()
Z = list(zip(X,Y))[:724]
random.shuffle(Z)
tol_len = len(Z)
train_len = int(tol_len*0.8)
train = Z[:train_len]
valid = Z[train_len:]
train_X = np.array([t[0] for t in train])
train_Y = np.array([t[1] for t in train])
valid_X = np.array([t[0] for t in valid])
valid_Y = np.array([t[1] for t in valid])
print(train_X.shape)
print(train_Y.shape)
print(valid_X.shape)
print(valid_Y.shape)
np.save('./data/train_X.npy',train_X)
np.save('./data/train_Y.npy',train_Y)
np.save('./data/valid_X.npy',valid_X)
np.save('./data/valid_Y.npy',valid_Y)
cls_num = [0]*6
for t in train_Y:
    cls_num[t]+=1
print(cls_num)
cls_num = [0]*6
for t in valid_Y:
    cls_num[t]+=1
print(cls_num)