"""
RNN介绍：
    全称为Recurrent neural network,循环神经网络，主要处理 序列数据的
    序列结构：后边数据对前边数据有依赖

    组成：
        词嵌入层
        循环网络层
        输出层

    词嵌入层： 把词（对应的索引）转换成词向量
    RNN层： 逐步处理词向量，生成 每个时间步的 隐藏状态
    全连接层： 通过线性变换将隐藏状态映射到输出，通常是1个词汇表中的概率分布
"""
import torch
import torch.nn as nn
import jieba

def dm01():
    text='北京冬奥会的进度条已经过半，不少外国运动员在完成自己比赛后踏上征途。'
    words = jieba.lcut(text)
    print(f'分词结果：{words}')
    embed=nn.Embedding(len(words),4)
    i=0
    for i,word in enumerate(words):  #enumerate() 返回列表中的每个值 及其对应的索引
        #把词索引转换成词向量
        word_vector=embed(torch.tensor(i))
        print(f'词:{word},\t\t词向量{word_vector}')
if __name__ == '__main__':
    dm01()