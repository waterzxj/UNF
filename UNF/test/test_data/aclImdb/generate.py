import os
import sys
import json
import random

label_pool = ["neg", "pos"]

for line in open(sys.argv[1]):
    line = line.rstrip()
    dic = {}
    dic["TEXT"] = line
    dic["LABEL"] = random.sample(label_pool, 1)[0]
    print(json.dumps(dic))
