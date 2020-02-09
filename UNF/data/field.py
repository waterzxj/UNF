#coding:utf-8
"""
对处理数据域的抽象
"""
from torchtext.data.field import RawField, Field, LabelField


class WordField(Field):
    """
    数据词域的抽象
    """
    def __init__(self, **kwarg):
        print(kwarg)
        super(WordField, self).__init__(**kwarg)

class CharField(Field):
    """
    数据字符域的抽象
    """
    def __init__(self, **kwarg):
        super(CharField, self).__init__(**kwarg)

class SiteField(Field):
    """
    站点域的抽象
    """
    def __init__(self,  **kwarg):
        super(SiteField, self).__init__(**kwarg)
