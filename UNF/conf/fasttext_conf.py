#coding:utf-8
"""
data_loader conf解释：
dataset: 配置训练数据和训练数据的格式，提供自动加载的功能
field: 针对每一个域提供一个相应的配置，每一个域的tokenzie，最大长度，是否返回padding长度（LSTM用）等
iterator: 提供迭代的配置，包括每个batch大小，device是cpu还是gpu
"""
#data_loader相关
data_loader_conf = {
    "dataset":{
        "path": "test/test_data/data",
        "train": "train_sample",
        "validation": "val_sample",
        "test": "test",
        "format": "json"
    },
    "fields":[{
        "name":"TEXT",
        "name_cls":"WordField",
        "attrs":{
            "tokenize":"WhitespaceTokenizer",
            }
        },
        {
            "name":"LABEL",
            "name_cls":"LabelField",
        }],
    "iterator":{
        "batch_size":512,
        "shuffle": True,
    }
}

#模型相关
model_conf = [
    {
        "name": "TEXT",
        "encoder_cls": "FastText",
        "encoder_params": {
            "input_dim": 100,
            "hidden_dim": 200,
        }
    }
]

#多域模型需要
aggregator_conf = {

}

decoder_conf = {
    
}

#learner相关的
learner_conf = {
    "num_epochs": 6,
    "optimizer": "Adam",
    "optimizer_parmas": {
        "lr": 1e-4
    },
    "device": "cuda:0",
    "loss": "CrossEntropyLoss",
    "serialization_dir": "sex_fasttext",
    "label_tag": "1",
    "use_fp16": True,
    "multi_gpu": True
}
