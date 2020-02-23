#!usr/bin/env python  
# -*- coding: utf-8 -*-  
import os
import sys
sys.path.append("..")
from flask import render_template, redirect, url_for, request, Blueprint
from collections import defaultdict
import random
import json

from models.predictor import Predictor

model_path = "../sex_textcnn3"
model_type = "TEXTCNN"

predictor = Predictor(model_path, model_type=model_type)

model_info = json.load(open("%s/conf.json" % model_path))
model_info["model_name"] = model_type
#just for beautiful
tmp = ""
for k,v in model_info.items():
    tmp += k
    tmp += ":"
    tmp += str(v)
    tmp += "  " 

tmp = tmp.rstrip()



static_location = 'static'
debug = Blueprint('debug', __name__, static_folder=static_location)
app = debug


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def similar():
    if request.method == 'GET':
        return render_template('web.html')
    else:
        title = request.form.get('title')
        score = predictor.predict(title)
        return render_template('web.html', score=score, input_value=title, model_info=tmp)

