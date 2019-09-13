`pip install keras-bert`

##### ./code/gen_text.py：
>split the sentences and write into file ./data人教+北师大中小学
    
##### ./data/train.txt&valid.txt
    enof sentence: "##",now useless

##### ./get_bert_CLS.py
>load **Bert** and predict CLS vector of each sentence

##### ./code/sample.py：
>sample train&valid(8:2) vectors from bert.npy,labels.npy 
>>train:[82, 82, 105, 101, 103, 106]
valid:[16, 35, 22, 25, 30, 17] 

##### ./data/
>***人教+北师大中小学:***
>>Max senteces numers:284
>>Max senteces length:220
>>sizes of each grade: [98, 117, 127, 126, 133, 123, 69, 68, 59]

>***人教+北师大小学:***
>>Max senteces numers:169
>>Max senteces length:153
---
##### on fine version of txtCNN
kl_size=[2,3,4],kl_dim = 50.
reach 0.47 - 0.50 acc on primary school dataset
