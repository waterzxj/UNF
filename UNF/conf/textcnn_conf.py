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
        "path":"test/test_data/aclImdb",
        "train":"train",
        "test":"test",
        "format":"json"
    },
    "fields":[{
        "name":"TEXT",
        "name_cls":"WordField",
        "attrs":{
            "tokenize":"WhitespaceTokenizer",
            "vectors": "path" #这个field是有词向量预训练好的
            }
        },
        {
            "name":"LABEL",
            "name_cls":"LabelField",
        }],
    "iterator":{
        "batch_size":64,
        "device":"cpu",
    }
}

#模型相关
model_conf = [
    {
        "name": "field_name",
        "encoder_cls": "TextCnn",
        "encoder_params": {
            "input_dim": 100,
            "filter_num": 100,
            "filter_size": [1,2,3,4],
            "dropout": 0.1,
            "pretrained": True,
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
    "device": "cuda:1",
    "loss": "CrossEntropyLoss",
    "serialization_dir": "model_save"
}

