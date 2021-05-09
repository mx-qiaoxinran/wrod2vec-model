# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:24:37 2021

@author: 86189
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from collections import Counter
import numpy as np
import random
import math
import pandas as pd
#import scipy
use_cuda = torch.cuda.is_available()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
#设置一些超参数
C = 3   #上下文单词为3个，窗口大小为3
K = 100  #负采样样本数
NUM_EPOCHS = 128
MAX_VOCAB_SIZE = 30000 #词汇表大小
BATCH_SIZE = 2
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100 #词向量大小

def word_tokenize(text):  #把一个语料库文本分割成一个一个的单词
    return text.split()
with open("text8.train.txt","r") as fin:
    text = fin.read()
#print(text[:1000].split())
text = text.split()#将文本分割成一个一个的单词
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))#(按照词频构建词典）找出该文本词频最高的前MAX_VOCAB_SIZE - 1个词，（有一个位置留给unk），并将其转换为一个词典
#该词典是（‘word’：该单词出现的次数）的一个结构
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))#将其余的单词（即除了最高的前MAX - 1的单词外）都设为unk，然后将个数设为unk词出现的次数

idx_to_word = [word for word in vocab.keys()]#存储词典所有单词的一个列表
word_to_idx = {word : i for i,word in enumerate(idx_to_word)}#存储（单词：单词的顺序索引）的一个词典（这个顺序索引用于后面对词进行编码）这些词是按照词频大小排序的

word_counts = np.array([count for count in vocab.values()],dtype = np.float32)#获取词典中每一个词的出现次数，并将其放在一个矩阵（向量）中
word_freqs = word_counts / np.sum(word_counts) #存储vocab中每一个单词的词频的向量
word_freqs = word_freqs ** (3./4.)
word_freqs =word_freqs / np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word)

class WordEmbeddingDataset(tud.Dataset):#把数据打包
    def __init__(self,text,word_to_idx,idx_to_word,word_freqs,word_counts):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded = [word_to_idx.get(word,word_to_idx["<unk>"]) for word in text] #(按照词频顺序进行编码）返回text中每一个词在词典word_to_idx中的顺序索引，如果不存在，返回<unk>的索引，这个列表可以表示为单词的编码，即可以用于后面one-hot编码
        self.text_encoded = torch.LongTensor(self.text_encoded) #将该编码列表转化为一个tensor
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
    def __len__(self):
        #这个数据集一共有多少个item
        return len(self.text_encoded)
    
    def __getitem__(self,idx):
        center_word = self.text_encoded[idx] #返回中心词，以idx作为索引的
        pos_indices = list(range(idx - C,idx)) + list(range(idx + 1,idx + C))#（上下文词汇）以idx为中心，左右两边取落在窗口内的词为上下文词汇
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]#类比循环队列，防止越界
        pos_words = self.text_encoded[pos_indices]#存储上下文词汇
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0],True)#（负采样）以word_freqs的权重（概率）在词汇表中取值（返回的是单词索引），取的次数为K * pos_words.shape[0],具体见收藏文档
        return center_word,pos_words,neg_words
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)#
dataloader = tud.DataLoader(dataset,BATCH_SIZE,shuffle=True)#将数据按照BATCH_SIZE进行打包，
    
#定义pytorch模型
class EmbeddingModel(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(EmbeddingModel,self).__init__()
        
        self.vocab_size = vocab_size #有多少个词
        self.embed_size = embed_size #每个词的词向量维度是多少
        
        self.in_embed = nn.Embedding(vocab_size, embed_size)#产生数量为vocab_size,维度大小为embed_size的词向量
        self.out_embed = nn.Embedding(vocab_size, embed_size)
        
        
    def forward(self,input_labels,pos_labels,neg_labels):
        #input_labels : [batch_size]
        #pos_labels :[batch_size,(window_size * 2)]
        #neg_labels :[batch_size,(window_size * 2 * K)]
        
        
        input_embedding = self.in_embed(input_labels)#将中心词词嵌入向量化，形状：[batch_size,embed_size]
        pos_embedding = self.out_embed(pos_labels)#将上下文词词嵌入向量化, 形状：[batch_size,(window_size * 2)，embed_size]
        neg_embedding = self.out_embed(neg_labels)#将负采样的词词嵌入向量化 形状：[batch_size,(window_size * 2 * K)，embed_size]
        
        #求损失函数
        input_embedding =input_embedding.unsqueeze(2)#[batch_size,embed_size，1]
        pos_dot = torch.bmm(pos_embedding,input_embedding).squeeze()
        neg_dot = torch.bmm(neg_embedding,-input_embedding).squeeze()
        
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        
        loss = log_pos + log_neg
        
        return -loss
    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()#返回所有的词向量，类型为array型

#定义一个模型并将其转移到GPU上
model = EmbeddingModel(VOCAB_SIZE,EMBEDDING_SIZE)
if use_cuda:
   model = model.cuda()


#训练模型
optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i,(input_labels,pos_labels,neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        
        if use_cuda:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
        optimizer.zero_grad()
        loss = model.forward(input_labels, pos_labels, neg_labels)
        loss.backward()
        optimizer.step()
            
        if i % 100 == 0:
            print("epoch",e,'iteration',i,loss.item())
        
        
    
        
        
        
        
        
        








