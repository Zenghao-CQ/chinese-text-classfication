./code/gen_text.py：
    split the sentences and write into new ./data/train.txt&valid.txt
    enof sentence: "##"
./code/gen_random.py：
    sample train.txt&valid.txt at 8:2
    train:
    [77, 90, 106, 103, 108, 93]->577
    valid:
    [21, 27, 21, 23, 25, 30]->147

./data/
    人教+北师大中小学:
    Max senteces numers:284
    Max senteces length:220
    sizes of each grade: [98, 117, 127, 126, 133, 123, 69, 68, 59]

    人教+北师大小学:
    Max senteces numers:169
    Max senteces length:153

fine verison of TextCNN kl-size=[2,3,4],kl_dim = 50.
reach 50% acc on _f.npy data, 47% acc on _new.npy in ./data/