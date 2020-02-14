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
        "path":"test/test_data/ner",
        "train":"train_ner",
        "test":"test_ner",
        "format":"json"
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
            "attrs":{
                "tokenize":"WhitespaceTokenizer",
                "sequential": True
            }
        }],
    "iterator":{
        "batch_size":64,
        "shuffle": True,
    }
}

#模型相关
model_conf = [
    {
        "name": "TEXT",
        "encoder_cls": "TextCnn",
        "encoder_params": {
            "input_dim": 100,
            "filter_num": 100,
            "filter_size": [1,2,3,4],
            "dropout": 0.1,
            "pretrained": False,
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
    "num_epochs": 10,
    "optimizer": "Adam",
    "optimizer_parmas": {
        "lr": 1e-5
    },
    "device": "cuda:1",
    "loss": "CrossEntropyLoss",
    "serialization_dir": "model_save"
}

