"""
文本处理的一般步骤
将非结构化自然语言转换成计算机可以理解的结构化数字数据
一、数据获取
二、数据清洗
    删除原始文本中与任务无关的冗余信息，修正错误数据，提高数据质量
    1、缺失值与重复值的处理
    2、特殊字符和格式的清理
    3、无关内容过滤
三、文本标准化
    将不同形式的同一表达统一为标准形式，降低模型学习难度
    1、大小写转换
    2、拼音纠错（pyspellchecker，TextBlob）
    3、词形归一化（英文）
        词干提取：去除词缀获得词根
        词形还原：基于词典还原为单词的基本形式（NLTK，spaCy）
    4、中文繁体转简体
四、文本分词(Tokenization)
    将连续的文本字符串切分为一个个独立的基本处理单元(Token)，这是文本向量化的前提
    1、字符级分词：将每个字符作为一个Token
    2、词级分词：将每一个单词作为一个Token(jieba)
    3、子词级分词：介于字符和词之间，将单词拆分为常见子词
五、构建词表(Vocabulary)
    建立Token与数字索引之间的双向映射
    标准流程：
        1、统计整个语料库中所有Token的出现频率
        2、过滤低频Token，减少词表大小和过拟合
        3、添加特殊Token:
            <unk>:未知词，对应训练集中未出现的Token
            <pad>:填充符，用于将不同长度的序列补全为相同长度
            <bos>/<eos>:序列开始/结束标记
        4、生成两个核心映射表
            token_to_idx: Token-->数字索引
            idx_to_token: 数字索引-->Token
六、文本向量化
    将分词后的Token序列转换为固定长度的数字向量，作为模型的输入
    主流方法：
        1、传统统计方法
            One-Hot编码：每个Token对应一个只有一位为1的稀疏向量
            词袋模型(BoW)：统计每个Token再文本中出现的次数
            TF-IDF：衡量Token对文本的重要程度（词频*逆文档频率）
        2、深度学习方法
            静态词嵌入(Word Embedding):每个Token对应一个固定的低维稠密向量
            上下文相关词嵌入：同一个词在不同语境下有不同的向量表示
        3、索引序列
            直接将Token转换为对应的词表索引，形成整数序列
            后续通过模型的嵌入层自动学习词向量
七、序列数据处理
    将长索引序列切分为固定长度的小序列，并按批量组织，满足模型要求
    标准操作：
        1、序列截断与填充
            超过指定长度的序列截断，不足的用<pad>索引填充
            保证所有序列长度一直，才能组成批量输入模型
        2、序列采样
            将长语料切分为固定长度的子序列，每个子序列作为一个训练样本
            语言模型的任务中，标签Y是输入X向右移动一个位置的序列
            两种主流的采样方式：
                随机抽样：每个批量的序列随机选取，相邻批量无连续性
                顺序分区：保证序列的连续性，相邻批量的序列首尾相连
八、数据划分与增强
    1、数据集划分（时序数据不能随机划分，需按时间顺序划分）
    2、文本数据增强
        扩充数据集，防止过拟合，提高模型泛化能力
            回译：将文本翻译成另一种语言再翻译回来
            同义词替换：随机替换文本中的部分词为同义词
            随即插入/删除：随机插入或删除少量Token
            语序打乱：随机打乱部分词的顺序
"""



import collections #统计词频
import re  #用于文本清洗的正则表达式
import random  #用于随机采样
import torch

#读取《时间机器》原始文本并进行标准化处理
def read_time_machine():
    with open('time_machine.txt', 'r', encoding="utf-8") as f:
        lines=f.readlines()
    #[^A-Za-z]正则表达式 匹配所有非英文字母的字符替换为空格
    #strip() 去除每行首位的空格
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]

#分词函数
def tokenize(lines,token='word'):
    if token=='word':
        return [line.split() for line in lines] #词级分词，语义级别，以单词之间的空格字符作为分隔符
    elif token=='char':
        return [list(line) for line in lines]  #字符级分词 将line中的所有字符转换为list
    else:
        print('错误：未知元类型:'+token)
#tokenize()函数输出的是一个大列表，其中的元素是每一行对应的单词或者字符[[token1,token2,...],[token1,token2,...],[...]]，是二维的

#词表构建，建立文本token与数字索引之间的双向映射
class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_word=None):
        #默认参数
        """
        为什么用None而不是直接[]？
        Python 中函数的可变默认参数在函数定义时创建，多次调用会共享同一个列表对象，导致意外的副作用。

        reserve_word:保留词，是在词表中预先定义、必须存在且拥有固定索引的特殊token
        如序列填充，未知词处理，序列边界标记
        """
        if tokens is None:
            tokens=[]
        if reserved_word is None:
            reserved_word=[]
        #统计所有token出现的频率
        counter=count_corpus(tokens)
        #lambda表达式中x[1]是指取列表或元组中的第二个元素排列，而x指代的是前面的counter.items(),reserve=True是指降序排列
        #由于counter是一个(token,频率)的元组，counter.items()将Counter内部的键值对映射结构，转换为可迭代的(token.频率)的二元序列组(元组)
        self._token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        #self.idx_to_token是按一定顺序的词或字符的列表
        self.idx_to_token=['<unk>']+reserved_word
        #token_to_index是一个字典，根据idx_to_token来翻译
        self.token_to_idx={token:idx
                           for idx,token in enumerate(self.idx_to_token)}
        for token,freq in self._token_freqs:
            #低频词过滤
            if freq<min_freq:
                """
                为什么用break而非continue
                因为self._token_freqs已经通过sorted排序严格从高到低，当它遇到第一个freq<min_freq的token时，后面的token的频率必然都小于等于它
                """
                break
            #避免添加重复存在的token，由频率从高到低
            if token not in self.token_to_idx:
                #先添加到idx_to_token列表，再同步添加到token_to_idx字典中，索引就是新添加元素的下标
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1

    def __len__(self):
        return len(self.idx_to_token)

    #重载[]运算符，支持单个token和批量token的转索引
    def __getitem__(self,tokens):
        #返回索引值
        #单个token：直接返回对应的索引，未知词返回<unk>的索引0
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        #批量token：递归调用，返回索引列表
        return [self.__getitem__(token) for token in tokens]

    #反向转换方法
    def to_token(self,indices):
        #单个索引：直接返回对应的token
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        #批量索引：返回token列表，递归调用
        return [self.idx_to_token[index] for index in indices]

    @property
    #<unk>永远排在词表最前面，保证其固定索引不变，保证模型的稳定性
    def unk(self):
        return 0

    #返回按频率降序排序的(token,频率)列表或元组
    def token_freq(self):
        return self._token_freqs

#词频统计函数，统计整个语料库中所有token的出现频率
#isinstance()函数，用来检查一个对象是不是某个类型的实例，且会考虑继承 isinstance(object,class)
def count_corpus(tokens):
    #处理空输入和自动展平二维token列表
    if len(tokens)==0 or isinstance(tokens[0],list):
        #将二维列表（每个token的列表）展平为一维
        tokens=[token for line in tokens for token in line]
    #collection.Counter()计数器统计频率,是字典子类，返回值是一个键对集合，输出的是一个字典
    return collections.Counter(tokens)

#整合上述所有函数，直接得到模型训练的所需数字语料库和相对应的词表对象
#max_token，限制使用的最大token数量，-1表示使用全部语料
def load_corpus_time_machine(max_token=-1):
    lines=read_time_machine()
    tokens=tokenize(lines,'char')
    vocab=Vocab(tokens)
    #将所有文本token转换为数字索引，生成一维语料库
    #调用重载后的__getitem__方法，将字符转换为对应的数字索引
    corpus=[vocab[token] for line in tokens for token in line]
    #可选截断语料到max_token
    if max_token>0:
        corpus=corpus[:max_token]
    return corpus,vocab
if __name__ == '__main__':
    # corpus表示文本转换后的一维索引列表
    corpus, vocab = load_corpus_time_machine()
    print(f"词表大小: {len(vocab)}, 语料长度: {len(corpus)}")

"""
模型的训练任务是：给定前面的num_steps个字符，预测第num_steps+1个字符
原始的语料库是一个超长的一维整数列表，需要
    1、切分为固定长度num_steps的小序列
    2、按batch_size组织成批量
    3、保证每个批量上的X ([batch_size,num_steps],每个元素是一个字符的索引) 与 Y (是X向右移动一个未知的序列) 满足对应关系
"""


def seq_data_iter_random(corpus,batch_size,num_steps):
    #随机偏移起始位置，增加训练的随机性从(0,num_steps-1)中选取
    corpus=corpus[random.randint(0,num_steps-1):]
    #计算可以切分的总子序列数
    #减1是因为每个子序列x需要对应y=x+1，最后一个字符不能作为x的开头
    num_subseqs=(len(corpus)-1)//num_steps
    #生成所有子序列的起始序列，然后打乱
    initial_indices=list(range(0,num_subseqs*num_steps,num_steps))
    random.shuffle(initial_indices)

    #返回从pos开始长度为num_steps的小序列
    def data(pos):
        return corpus[pos:pos+num_steps]


    #总批次，每个批次包含batch_size个包含num_steps个的整数序列
    num_batches=num_subseqs//batch_size
    #逐批量生成数据
    for i in range(0,batch_size*num_batches,batch_size):
        #取出当前批量的batch_size个随机起始索引
        initial_indices_per_batch=initial_indices[i:i+batch_size]
        #得到输入X和标签Y
        X=[data(j) for j in initial_indices_per_batch]
        Y=[data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X),torch.tensor(Y)

#相邻批量的序列是连续的，因此可以在批量之间传递 RNN 的隐藏状态，让模型能够学习更长距离的依赖关系
#将整个语料库按行切分为batch_size个连续的长序列，然后切分为固定长度的小序列，同一个批量中不同行的序列是连续的，相邻批量的序列也是连续的
def seq_data_iter_sequential(corpus,batch_size,num_steps):
    #随机偏移起始位置
    offset=random.randint(0,num_steps)
    #计算可以使用的token数，保证能被batch_size整除
    #减1同样是因为X+1=Y
    num_tokens=((len(corpus)-offset-1)//batch_size)*batch_size
    #生成X输入和Y标签
    Xs=torch.tensor(corpus[offset:offset+num_tokens])
    Ys=torch.tensor(corpus[offset+1:offset+num_tokens+1])
    #将长序列reshape成batch_size行，每行是一个连续的长序列，前面的整除发挥用处
    Xs,Ys=Xs.reshape(batch_size,-1),Ys.reshape(batch_size,-1)
    #计算总批次数（总列数除以num_steps）
    num_batches=Xs.shape[1]//num_steps
    for i in range(0,num_steps*num_batches,num_steps):
        X=Xs[:,i:i+num_steps]
        Y=Ys[:,i:i+num_steps]
        yield X,Y

#封装两种迭代器
class SeqDataLoader:
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens):
        if use_random_iter:
            self.data_iter_fn=seq_data_iter_random
        else:
            self.data_iter_fn=seq_data_iter_sequential
        self.corpus,self.vocab=load_corpus_time_machine(max_tokens)
        self.batch_size,self.num_steps=batch_size,num_steps
    def __iter__(self):
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)

def load_data_time_machine(batch_size,num_steps,use_random_iter=False,max_tokens=10000):
    data_iter=SeqDataLoader(batch_size,num_steps,use_random_iter,max_tokens)
    return data_iter,data_iter.vocab