# Introduction
#### Designing philosophy
- ***modularity***: specifically, on the one hand, it is convenient to quickly run some nlp-related tasks; on the other hand, it is convenient for secondary development and research to implement some new models or technologies.
- ***efficiency***: supports distributed training and half-precision training, which is convenient for quickly training the model, although the current support is relatively crude
- ***comprehensive***: support pytorch trace into static graph, support c ++ server, provide web-server for debugging tools

# Support Tasks
Now, support ***text classification*** and ***sequeence labeling*** related tasks. In the future, will add text generate and text match related tasks.

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
![image](https://github.com/waterzxj/UNF/blob/master/readme/system.png)

# Module relation
![image](https://github.com/waterzxj/UNF/blob/master/readme/module.png)



