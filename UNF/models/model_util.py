#coding:utf-8
import  json

class Config(object):
    @classmethod
    def from_dict(cls, json_object):
        config = cls()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))
