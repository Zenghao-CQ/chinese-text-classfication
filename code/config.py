import os
epoch_num = 50
sent_num = 176#nums of one squence 169
              #Max senteces length:153,useless
batch_size = 15
learning_rate = 0.0005
vec_dim = 768#dim of wordvector
lstm_dim = 100
class_num = 6
data_path = os.path.join((os.path.dirname(os.path.dirname(__file__))),'data')
model_path = os.path.join(os.path.dirname(data_path),'model')
bert_path =  "/home/alex/bert_cls/chinese_L-12_H-768_A-12"