#coding:utf-8
"""
对数据加载的抽象，从磁盘数据load成模型可训练的数据格式
"""
import random

from torchtext.data import Field, Dataset, LabelField, TabularDataset
from torchtext.data import Iterator, BucketIterator

from data.field import WordField, CharField, SiteField
from data.tokenizer import BaseTokenizer, WhitespaceTokenizer, SpacyTokenizer


class DataLoader(object):
    """
    流程步骤：
    1）创建Field，每个类型的字段生成一个Field；生成Field的主要参数包括 tokenize等
    2）根据上一步创建的Field和数据路径，创建Dataset对象，Dataset就是Example对象的集合
    3）根据上一步创建的Dataset对象，创建Iterator对象，Iterator就是提供一个迭代方法，懒加载的返回
    一个Batch对象，生成Batch对象的时候会调用Field对象的process方法完成string2id的映射转换和pad过程；
    Batch对象主要是包含一个batch_size的Example对象

    """

    def __init__(self, config):
        self.config = config
        self.SEED = 1441 #magic data
        self.fields = None

    def generate_dataset(self):
        fields = self.config["fields"]
        #step1: Field生成
        inner_fields = {}
        inner_info = {}

        for item in fields:
            f_name = item["name"]
            f_cls = item["name_cls"]
            if "attrs" in item:
                if "tokenize" in item["attrs"]:
                    #初始化field的tokenizer对象
                    if "language" in item["attrs"]:
                        item["attrs"]["tokenize"] = globals()[item["attrs"]["tokenize"]](item["attrs"]["language"])
                    else:
                        item["attrs"]["tokenize"] = globals()[item["attrs"]["tokenize"]]()

                    #初始化field建词表的信息 
                    if "min_count" in item["attrs"]:
                        inner_info[f_name] = item["attrs"]["min_count"]
                        del item["attrs"]["min_count"]


                inner_fields[f_name] = (f_name, globals()[f_cls](**item["attrs"]))
            else:
                inner_fields[f_name] = (f_name, globals()[f_cls]())
                
        self.fields = inner_fields

        #step2: Dataset生成
        datasets = TabularDataset.splits(**self.config["dataset"] , fields=inner_fields)
        if len(datasets) == 2:
            #训练集、验证集的划分
            train_datasets, test_datasets = datasets
            train_datasets, valid_datasets = train_datasets.split(random_state=random.seed(self.SEED))
        elif len(datasets) == 3:
            train_datasets, valid_datasets, test_datasets = datasets

        #step3: 根据Field生成对应的词表
        for item in inner_fields.values():
            name, obj = item
            if obj.use_vocab:
                if name in inner_info:
                    obj.build_vocab(datasets[0], min_freq=inner_info[name])
                else:
                    obj.build_vocab(datasets[0])

        #step4: Iterator对象生成
        data_iterator = BucketIterator.splits((train_datasets, valid_datasets, test_datasets), sort=False, **self.config["iterator"])

        return data_iterator
