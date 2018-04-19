#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:44:46 2018

@author: liangyangxiao
"""

from nltk.stem.porter import *
import collections
import nltk.classify.util, nltk.metrics
from nltk import precision
from nltk import recall
import nltk.classify
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import pandas as pd
import string
import nltk

text=''
def sentiment_test(str):
    global text
    text=stemwordlist(str)

file=pd.read_csv("imdb_labelled.txt",sep='\t',names=['txt','liked'])
#reviews = list(csv.reader(file))
print(file.head())

#获取file中“txt”并转为list
txt_list=list(file.txt)
liked_list=list(file.liked)

neg_sentence_list=[]
pos_sentence_list=[]

def get_tokens(text):
    lowers = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stopset = set(stopwords.words('english')) - set(( 'more', 'most', 'no', 'not', 'only', 'few', 'so', 'too', 'very', 'just', 'any'))

def stemwordlist(text):
    tokens = get_tokens(text)#改成循环
    filtered = [w for w in tokens if not w in stopset]
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    return stemmed

for i in range(0,len(liked_list)):
    if liked_list[i]==0:
        neg_sentence_list.append(stemwordlist(txt_list[i])) 
    else:
        pos_sentence_list.append(stemwordlist(txt_list[i]))
print('')
print(pos_sentence_list[0])
print('')
print('The model has been built successfully!')
def evaluate_classifier(features):
  
    negfeats = [(features(f), 'neg') for f in neg_sentence_list]
    posfeats = [(features(f), 'pos') for f in pos_sentence_list]
    
    negcutoff = len(negfeats)*9//10
    poscutoff = len(posfeats)*9//10
 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
 
    classifier = nltk.classify.SklearnClassifier(LinearSVC())
    classifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
    
#    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
#    print('pos precision:', precision(refsets['pos'], testsets['pos']))
#    print('pos recall:', recall(refsets['pos'], testsets['pos']))
#    print('neg precision:', precision(refsets['neg'], testsets['neg']))
#    print('neg recall:', recall(refsets['neg'], testsets['neg']))
    return classifier.classify(features(text))
 
def word_feats(words):
    return dict([(word, True) for word in words])

def single_word_features():
    print('evaluating single word features')
    print('')
    return evaluate_classifier(word_feats)
    
    
##################
word_fd = FreqDist()  #Create a new list to store the number of occurrences of all the words.
label_word_fd = ConditionalFreqDist()#Create a new list to store the number of occurrences of all the words with conditions.

for f in pos_sentence_list:
    for word in f:
        word_fd[word] += 1   #After receiving the parameter 'words', the frequency of each' word 'in' words' will be counted and a dictionary will be returned. Key is' word ', and value is the number of occurrences of word in words.
        label_word_fd['pos'][word] += 1

for f in neg_sentence_list:
    for word in f:
        word_fd[word] += 1
        label_word_fd['neg'][word] += 1
 

pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count
 
word_scores = {}
 
for word, freq in word_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
        (freq, pos_word_count), total_word_count)   #用BigramAssocMeasures.chi_sq函数为词汇计算pos评分
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score



best = sorted(word_scores.items(),key=lambda s: s[1],reverse=True)[:1000]

bestwords = set([w for w, s in best])

def high_information_feats(words):
    return dict([(word, True) for word in words if word in bestwords])

def high_information_features():
    print('evaluating best word features')
    print('')
    return evaluate_classifier(high_information_feats)
    

def high_information_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    
    bigram_finder = BigramCollocationFinder.from_words(words)
    try:bigrams = bigram_finder.nbest(score_fn, n)
    except:bigrams=[]
    d = dict([(bigram[0]+bigram[1], True) for bigram in bigrams])
    d.update(high_information_feats(words))
    return d
    
     
#print('evaluating best words + bigram chi_sq word features')
def high_information_bigram_features():
    return evaluate_classifier(high_information_bigram_word_feats)