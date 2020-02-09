from unittest import TestCase

from UNF.data.data_loader import DataLoader

class TestDataLoader(TestCase):

    def test_data_loader(self):
        config = {
                    "dataset": {
                        "fields": [{
                            "name": "",
                            "cls": "WordField"
                            "attrs": {
                                "tokenize": "WhitespaceTokenizer",
                            }           
                        },
                        {
                            "name": "",
                            "cls": "LabelField",    
                        }],
                        "dataset": {
                            "path": "",
                            "train": "",
                            "validation": "",
                            "test": "",
                            "format":""            
                        },
                        "iteration": {
                            "batch_size": 64,
                            "device": "cpu"
                        }
                    }
                }
        data_loader = DataLoader(config)
        print(data_loader)
        

