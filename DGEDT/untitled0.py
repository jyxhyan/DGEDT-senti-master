# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:41:06 2019

@author: tomson
"""

import numpy as np
import spacy
import pickle
from stanfordcorenlp import StanfordCoreNLP
import json
#from data_utils import *
#dataset='twitter'
#tmp=pickle.load(open(dataset+'_datas.pkl', 'rb'))
#class DepParser:
#    def __init__(self, url):
#        self.nlp = StanfordCoreNLP(url)
#        # Do not split the hyphenation
#        self.nlp_properties = {
#            'annotators': "depparse",
#            "tokenize.options": "splitHyphenated=false,normalizeParentheses=false",
#            "tokenize.whitespace": True,  # all tokens have been tokenized before
#            'ssplit.isOneSentence': True,
#            'outputFormat': 'json'
#        }
#
#    def get_head(self, text):
#        raw_text = " ".join(text)
#        parsed = self.nlp.annotate(raw_text.strip(), self.nlp_properties)['sentences'][0]['basicDependencies']
#        assert len(text) == len(parsed), "Lengths are note equal."
#        dep_head = len(parsed) * [0]
#        # Get the head node of each token
#        for p in parsed:
#            index = p['dependent'] - 1  # avoid ROOT
#            head = p['governor'] - 1  # start from 0
#            dep_head[index] = head
#        return dep_head
#
#    def __call__(self, text):
#        return self.get_head(text)
#from spacy.tokenizer import Tokenizer
#from spacy.lang.en import English
#nlp = English()
## Create a blank Tokenizer with just the English vocab
#tokenizer = Tokenizer(nlp.vocab)
#
## Construction 2
#from spacy.lang.en import English
#nlp = English()
## Create a Tokenizer with the default settings for English
## including punctuation rules and exceptions
#tokenizer = nlp.Defaults.create_tokenizer(nlp)    
#tokens = tokenizer("This is  a sentence . fg")
#print(list(tokens))
#import re
#print(re.sub(r' {2,}',' ','a   n'))
#nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27')
#text = "a 're tom. jk jk."
#parsed=None
#import re
##print ('Tokenize:', nlp.word_tokenize(text))
#def tokenizeanddep(text):
#    text+=' '
#    text=re.sub(r'\. ',' . ',text).strip()
##    print(text)
#    text=re.sub(r' {2,}',' ',text)
#    nlp_properties = {
#                'annotators': "depparse",#tokenize
##                "tokenize.options": "splitHyphenated=false,normalizeParentheses=false,strictTreebank3=true,asciiQuotes=true,latexQuotes=true,ptb3Escaping=true",
#                "tokenize.whitespace": True,  # all tokens have been tokenized before
#                'ssplit.isOneSentence': False,
#                'outputFormat': 'json'
#            }
#    global parsed
#    parsed = json.loads(nlp.annotate(text.strip(), nlp_properties))
#    parsed=parsed['sentences']
#    tokens=[]
#    tuples=[]
#    tmplen=0
#    for item in parsed:
#        tokens.extend([ite['word'] for ite in item['tokens']])
#        
#        tuples.extend( [(ite['dep'],ite['governor']-1+tmplen,ite['dependent']-1+tmplen) for ite in item['basicDependencies'] if ite['dep']!='ROOT'])
#        tmplen+=len(tokens)
#    return tokens,tuples
#a=tokenizeanddep(text)
#nlp.close()
#from stanfordcorenlp import StanfordCoreNLP


#sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
#print ('Dependency Parsing:', nlp.dependency_parse(text))
#['sentences'][0]['basicDependencies']
#for p in parsed:
#    print (p.keys())
#assert len(text) == len(parsed), "Lengths are note equal."
nlp = spacy.load('en_core_web_sm')
text='Guangdong University of Foreign Studies is located in Guangzhou.'
##a=spacy.tokenizer(text)
document = nlp(text)
print([t.dep_ for t in document])
print(document)
for token in document:
    print(token.dep_,list(token.children),token.text,token.i,token.head)