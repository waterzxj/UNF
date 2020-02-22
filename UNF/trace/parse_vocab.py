#coding:utf-8
import os
import sys
import json

vocab = json.load(open("vocab.txt"))
for item in vocab:
    print(item)

