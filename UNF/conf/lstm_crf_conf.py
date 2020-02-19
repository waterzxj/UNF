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
        "train":"train_ner_sample",
        "test":"test_ner_sample",
        "format":"json"
    },
    "fields":[{
        "name":"TEXT",
        "name_cls":"WordField",
        "attrs":{
            "tokenize":"WhitespaceTokenizer",
            "include_lengths": True
            }
        },
        {
            "name":"LABEL",
            "name_cls":"Field",
            "attrs":{
                "tokenize":"WhitespaceTokenizer",
                "sequential": True,
                "unk_token": None
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
        "encoder_cls": "LstmCrfTagger",
        "encoder_params": {
            "input_dim": 100,
            "hidden_size": 200,
            "num_layers": 3,
            "device": "cuda:1"
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
    "serialization_dir": "model_lstm_example",
    "sequence_model": True,
    "metric": "NerF1Measure"
}

