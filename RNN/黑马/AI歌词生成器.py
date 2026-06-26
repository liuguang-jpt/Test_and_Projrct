"""
RNN案例：
    基于杰伦歌词来训练模型，用给定的起始词，结合长度，来进行AI歌词生成

实现步骤：
    1、获取数据，进行分词，获取词表
    2、数据预处理，构建数据集
    3、搭建RNN神经网络
    4、模型训练
    5、模型测试
"""
import torch
import torch.nn as nn
import torch.optim as optim
import jieba
from torch.utils.data import Dataset, DataLoader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_vocab():
    #定义变量：记录去重后的所有的词，每行文本的分词结果
    unique_words,all_words=[],[]
    for line in  open("jaychou_lyrics.txt", encoding="utf-8"):
        #获取每行歌词，进行分词
        words=jieba.lcut(line.strip())
        #所有分词结果记录到 all_words 中
        all_words.append(words)
        #遍历分词结果，去重后添加到 unique_words 中
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    #统计语料中（去重后）词的数量
    unique_word_count=len(unique_words)
    index_to_word={i:word for i,word in enumerate(unique_words)}
    word_to_index={word:i for i,word in enumerate(unique_words)}
    #歌词文本用词表索引表示
    corpus_idx=[]
    for words in all_words:
        for word in words:
            corpus_idx.append(word_to_index[word])
    #返回结果：唯一词列表，词表，去重后词的数量，歌词文本用词表索引表示
    return unique_words,word_to_index,index_to_word,unique_word_count,corpus_idx

class lyricsDataset(Dataset):
    #初始化词索引，词个数等
    def __init__(self,num_chars,corpus_idx):
        #文档数据中词的索引
        self.corpus_idx=corpus_idx
        #每个句子中词的个数
        self.num_chars=num_chars
        #文档数据中，词的数量（不去重）
        self.total_len=len(corpus_idx)
        #句子数量
        self.number=self.total_len // self.num_chars

    def __len__(self):
        #返回句子数量
        return self.number

    def __getitem__(self, index):

        x=self.corpus_idx[index:index+self.num_chars]
        y=self.corpus_idx[index+1:index+self.num_chars+1]
        return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)

class TextGenerator(nn.Module):
    def __init__(self,unique_word_count):
        super().__init__()
        #初始化词嵌入层，语料中词的数量，词向量的维度
        self.ebd=nn.Embedding(unique_word_count,128)
        #循环网络层，输入维度（词向量维度），输出维度（隐藏层维度），网络层数
        self.rnn=nn.RNN(128,256,1)
        #输出层:特征向量的维度（和隐藏层向量维度一致），词表中词的个数
        self.out=nn.Linear(256,unique_word_count)

    def forward(self,inputs,hidden):
        ebd=self.ebd(inputs)
        output,hidden=self.rnn(ebd.transpose(0,1),hidden)
        output=self.out(output.reshape(-1,output.shape[-1]))
        return output,hidden

    def init_hidden(self,batch_size):
        return torch.zeros(1,batch_size,256,device=device)

def train(lyrics_dataset,unique_word_count):

    model=TextGenerator(unique_word_count).to(device)
    lyrics_dataloader=DataLoader(lyrics_dataset,batch_size=8,shuffle=True)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    epochs=200

    model.train()
    for epoch in range(epochs):
        strat=time.time()
        iter_num=0
        total_loss=0
        for x,y in lyrics_dataloader:
            x,y=x.to(device),y.to(device)
            hidden=model.init_hidden(batch_size=x.size(0))

            output,hidden=model(x,hidden)
            y=y.transpose(0,1).reshape(-1)

            loss=criterion(output,y)
            total_loss+=loss.item()
            iter_num+=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch {epoch+1} loss: {total_loss/iter_num} time: {time.time()-strat}')
    torch.save(model.state_dict(), '../model.pth')

def evaluate(start_word,sentence_length,word_to_index,index_to_word,unique_word_count,corpus_idx):

    model=TextGenerator(unique_word_count).to(device)
    model.load_state_dict(torch.load('../model.pth'))
    model.eval()

    hidden=model.init_hidden(batch_size=1)

    current_idx=torch.tensor([[word_to_index[start_word]]],device=device,dtype=torch.long)
    generated=[start_word]

    with torch.no_grad():
        for _ in range(sentence_length):
            output,hidden=model(current_idx,hidden)

            current_idx=torch.argmax(output,dim=-1).unsqueeze(0)

            next_word=index_to_word[current_idx.item()]
            generated.append(next_word)

    print(' '.join(generated))


if __name__ == '__main__':
    unique_words, word_to_index, index_to_word, unique_word_count, corpus_idx = build_vocab()
    print(f'词表大小：{unique_words}')
    lyrics_dataset = lyricsDataset(num_chars=32, corpus_idx=corpus_idx)
    #train(lyrics_dataset,unique_word_count)
    evaluate(
        start_word='晴天',
        sentence_length=50,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        unique_word_count=unique_word_count,
        corpus_idx=corpus_idx
    )




