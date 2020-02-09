#coding:utf-8
"""
对分词的抽象
"""

class BaseTokenizer(object):
    """
    Tokenize的基类
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return self.tokenize(x)

    def tokenize(self, x):
        raise Exception("Not Implemented!")


class WhitespaceTokenizer(BaseTokenizer):
    """
    空格切分
    """
    def __init__(self):
        super(WhitespaceTokenizer, self).__init__()

    def tokenize(self, x):
        return x.split()


class SpacyTokenizer(BaseTokenizer):
    """
    spacy切分
    """
    def __init__(self, language):
        super(SpacyTokenizer, self).__init__()
        self.language = language
        self.init()

    def init(self):
        import spacy
        self.spacy = spacy.load(self.language)

    def tokenize(self, x):
        return [tok.text for tok in self.spacy.tokenizer(x)]


        
if __name__ == "__main__":
    print(WhitespaceTokenizer("a b c"))
