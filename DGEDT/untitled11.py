# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:14:31 2019

@author: tomson
"""

import numpy as np
import spacy
import copy
from spacy import displacy
def diji(adj):
    adj=np.array(adj)
    adjs=copy.deepcopy(adj)
    adjs[adjs==0]=1000
    length=adj.shape[0]
    for u in range(length):
        for i in range(length):
            for j in range(length):
                if adjs[i,u]+adjs[u,j]<adjs[i,j]:
                    adjs[i,j]=adjs[i,u]+adjs[u,j]
    adjs=(1/adjs)**1
    print(adjs)
    adjss=adjs.sum(-1,keepdims=True)
    print(adjss)
    adjs=adjs/adjss
    return adjs
a=diji([[1,1,0],[1,1,1],[0,1,1]])
#nlp = spacy.load('en_core_web_sm')
##nlp = spacy.load('en')
#doc = nlp( "It has a bad memory but a great battery life ." )
##a=spacy.tokenizer("It has a lot of memory and a great battery life .")
#print([t.dep_ for t in doc])
#displacy.serve(doc, style='dep',options={'distance':120,'compact':True})
#import spacy
#from spacy import displacy
#from pathlib import Path
#
#nlp = spacy.load("en_core_web_sm")
#sentences = ["It has a bad memory but a great battery life ."]
#for sent in sentences:
#    doc = nlp(sent)
#    svg = displacy.render(doc, style="dep", jupyter=False,options={'distance':140})
#    file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".jpg"
#    output_path = Path("images/" + file_name)
#    output_path.open("w", encoding="utf-8").write(svg)
#a='我修改订单了啊，然后系统没改过来呢<s>距离元吗亲爱的<s>不远<s>无需修改的亲爱的<s>我就是名字搞错了<s>到时候联系配送就可以啦亲爱的'
#b='要不每个订单一张发票太零散了<s>这个是随商品一起寄出的哈<s>额，都是同一个类别的商品嘛<s>不能呢 一个订单一张发票<s>好吧<s>请问还有其他还可以帮到您的吗?'
#a=a.split('<s>')
#b=b.split('<s>')
#print('load model successfully')
#print('load vectors and initialized')
#for idx,item in enumerate(b):
#    if idx%2==0:
#        print('input:')
#        print(item)
#    else:
#        print('response:')
#        print(item)
#c=input('input:')
#import torch
#import torch.nn as nn
#class selfalignment(torch.nn.Module):
#    """
#    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#    """
#    def __init__(self, in_features, bias=False):
#        super(selfalignment, self).__init__()
#        self.in_features = in_features
#        self.dropout = nn.Dropout(0.3)
##        self.out_features = out_features
#        self.weight = nn.Parameter(torch.FloatTensor(in_features, in_features))
#        self.weight1 = nn.Parameter(torch.FloatTensor(in_features, in_features))
#        self.linear=torch.nn.Linear(in_features, in_features,bias=False)
#        if bias:
#            self.bias = nn.Parameter(torch.FloatTensor(in_features))
#        else:
#            self.register_parameter('bias', None)
#
#    def forward(self, text, text1, textmask):
#        logits=torch.matmul(self.linear(text),text1.transpose(1,2))
#        masked=textmask.unsqueeze(1)
#        masked=(1-masked)*-1e20
#        logits=torch.softmax(logits+masked,-1)
#        output = torch.matmul(logits,text1)+text
#        output=output*textmask.unsqueeze(-1)
#        if self.bias is not None:
#            return self.dropout(output + self.bias),logits*textmask.unsqueeze(-1)
#        else:
#            return self.dropout(output),logits*textmask.unsqueeze(-1)
#a=selfalignment(112)
##torch.save(a.state_dict(),'hj.pt')
#a.load_state_dict(torch.load('hj.pt'),strict=False)