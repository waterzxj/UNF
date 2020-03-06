# Introduction

UNF(Universal NLP Framework) is built on pytorch and torchtext. Its design philosophy is：
- ***modularity***: specifically, on the one hand, it is convenient to quickly run some nlp-related tasks; on the other hand, it is convenient for secondary development and research to implement some new models or technologies.
- ***efficiency***: supports **distributed training** and **half-precision** training, which is convenient for quickly training the model, although the current support is relatively crude
- ***comprehensive***: support pytorch **trace** into static graph, support **c ++** server, provide web-server for **debugging tools**

# Support Tasks
Now, support ***text classification*** and ***sequence labeling*** related tasks. 

# Related Papers
- Convolutional Neural Networks for Sentence Classification [2014](https://arxiv.org/abs/1408.5882)
- Bag of Tricks for Efficient Text Classification [2016](https://arxiv.org/pdf/1607.01759.pdf)
- Deep Pyramid Convolutional Neural Networks for Text Categorization [2017, ACL](https://www.aclweb.org/anthology/P17-1052)
- Hierarchical Attention Networks for Document Classification [2017, ACL](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
- A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING [2017,ICLR](https://arxiv.org/abs/1703.03130)
- Joint Embedding of Words and Labels for Text Classification[2018,ACL](https://www.aclweb.org/anthology/P18-1216/)
- Neural Architectures for Named Entity [2016,ACL](https://www.aclweb.org/anthology/N16-1030/)
- Semi-supervised Multitask Learning for Sequence Labeling [2017, ACL](https://arxiv.org/abs/1704.07156)


# Framwork
![image](https://github.com/waterzxj/UNF/blob/master/pic/system.png)


# Module relation

Module name | Module function
---|---
 UNF.data  | Load data from disk to RAM, include batch, padding,numerical
UNF.module  | Neural network layer, include encoder, decoder, embedding, provided for use by the model
UNF.model | Neural network model structure, include DpCnn, SelAttention,Lstm-crf..and python predictor for those models
UNF.training | Model training, include early stopping, model save and reload, visualize metrics throuth Tensorboard
UNF.tracing | Trace pytorch dynamic graph to static graph, and provide c++ serving
UNF.web_server | Web server tool related


# Requirement
python3

pip3 install -r requirement.txt

# Training

```
#quick start
python3 train_flow.py
```
***Only* 5 line code need**
```
#data loader
data_loader = DataLoader(data_loader_conf)
train_iter, dev_iter, test_iter = data_loader.generate_dataset()

#model loader
model, model_conf = ModelLoader.from_params(model_conf, data_loader.fields)

#learner loader
learner = LearnerLoader.from_params(model, train_iter, dev_iter, learner_conf, test_iter=test_iter, fields=data_loader.fields, model_conf=model_conf)

#learning
learner.learn()
```
### 代码库提供了train_flow.py脚本可直接开箱运行示例

### 如下conf设置即可跑混合精度训练和多gpu训练
```
"use_fp16": False,
"multi_gpu": False
```

### 训练结果自动注入tensorboard监控
![image](https://github.com/waterzxj/UNF/blob/master/pic/tensorboard1.png)
![image](https://github.com/waterzxj/UNF/blob/master/pic/tensorboard2.png)

# Python inference

```
#quick start
python3 score_flow.py
```

```
#core code
from models.predictor import Predictor

predictor = Predictor(model_path, device, model_type)
logits = predictor.predict(input)

(0.18, -0.67)
```

# C++ inference

### step1: Trace dynamic graph to static graph


```
#quick start
python3 trace.py
```

```
#core code
net = globals()[model_cls](**config.__dict__)
net.load_state_dict_trace(torch.load("%s/best.th" % model_path))
net.eval()

mock_input = net.mock_input_data()
tr = torch.jit.trace(net, mock_input)
tr.save("trace/%s" % save_path)
```

### step2: c++ serving
- install cmake
- download [libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.2.0.zip) and unzip to trace folder

```
cd trace
cmake -DCMAKE_PREFIX_PATH=libtorch .
```
![image](https://github.com/waterzxj/UNF/blob/master/pic/cmake.png)

```
make
```
![image](https://github.com/waterzxj/UNF/blob/master/pic/make.png)

```
./predict trace.pt predict_vocab.txt
output: 2.2128 -2.3287
```

# RESTFUL-API web demo

```
cd web_server
python run.py
```

![image](https://github.com/waterzxj/UNF/blob/master/pic/web_demo.png)

