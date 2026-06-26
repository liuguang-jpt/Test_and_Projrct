import math
import time
import torch
import torch.nn as nn
from The_Time_Machine.The_Time_Machine import load_data_time_machine

"""
RNN:通过隐藏状态保存序列的历史信，实现对时序数据的建模
自回归生成:模型每次生成一个字符，然后用这个字符作为下一次输入，逐字符生成文本
梯度裁剪:解决RNN训练中梯度爆炸的问题
困惑度:语言模型的评估指标，值越小表示模型预测越准确

有隐状态的循环神经网络：
    当前时间步的隐藏变量由当前时间步的输入和前一个时间步的隐藏变量共同计算得出
    隐藏状态保存了序列直到当前时间步的历史信息-->隐状态
    
在训练模式下，上一次的输出不会作为这一次的输入，每个时间步的输入X_t都是语料库中的真实字符
在生成模式下，自回归生成
但是RNN的隐状态永远依赖于当前输入和上一个隐状态
"""

#参数初始化
def get_params(vocab_size,num_hiddens,device):
    #字符级模型：输入是one-hot向量，输出是每个字符的概率，输入输出维度都是词表大小
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01

    #隐藏层参数
    W_xh=normal((num_inputs,num_hiddens))  #输入->隐藏层权重
    W_hh=normal((num_hiddens,num_hiddens))  #上一时刻隐藏状态->当前上课隐藏状态权重
    b_h=torch.zeros(num_hiddens,device=device)  #隐藏层偏置

    #输出层参数
    W_hq=normal((num_hiddens,num_outputs))  #隐藏层->输出层权重
    b_q=torch.zeros(num_outputs,device=device)  #输出层偏置

    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size,num_hiddens,device):
    #torch.zeros((batch_size,num_hiddens),device=device):生成一个形状为[批量大小，隐藏层维度]的全零张量，是RNN的初始隐藏状态
    #外面的()：将这个张量包裹成一个元组，最后的,解释其是包含单个元素的元组
    #循环神经网络的标准接口设计，普通RNN只有一个隐藏状态，但是LSTM由两个隐藏状态：隐藏状态H和细胞状态C
    return (torch.zeros((batch_size,num_hiddens),device=device),)

#前向传播
def rnn(inputs,state,params):
    #inputs的形状为(时间步数，批量大小，词表大小)，时间步数是RNN一次向前传播能“看到”的最大单词数量，每个训练样本的固定长度
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state #隐藏状态
    outputs=[]
    #根据时间步数连续，生成(批量大小，词表大小)
    for X in inputs:
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)
    #将所有时间步的输出拼接：[num_steps*batch_size,vocab_size]，按照列拼接
    #将当前最后时刻的隐藏状态 H 包裹成元组,统一接口（兼容 LSTM 双隐态格式）,供下一轮计算继续传递使用
    return torch.cat(outputs,0),(H,)

#将三个函数全部封装到类中
class RNNModelScratch:
    """
    forward_fn
        本质：传入的RNN 前向传播函数（本例就是自定义的 rnn 函数）
        作用：封装整套时序计算逻辑，负责根据输入序列、上一时刻隐藏状态、模型参数，计算当前输出和最新隐藏状态。
        调用时机：RNNModelScratch 实例被调用时（__call__），内部执行 self.forward_fn 完成前向计算。
        设计目的：解耦模型框架与具体前向逻辑，替换 LSTM/GRU 时只需更换该函数，主体代码不用改动。
    """
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens=vocab_size,num_hiddens
        self.params=get_params(vocab_size,num_hiddens,device)
        self.init_state,self.forward_fn=init_state,forward_fn

    def __call__(self,X,state):
        #输入X的形状[batch_size,num_steps]转置后方便按照时间步循环
        #one-hot编码：[num_steps,batch_size,vocab_size]
        X=nn.functional.one_hot(X.T,self.vocab_size).type(torch.float32)
        return self.forward_fn(X,state,self.params)

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

#prefix:原字符，num_preds:预测字符的个数
def predict_ch8(prefix,num_preds,net,vocab,device):
    #生成初始隐藏状态，batch_size=1，表示对单个字符做初始化
    state=net.begin_state(batch_size=1,device=device)
    #把每个字符在vocab的对应整型的下标
    outputs=[vocab[prefix[0]]]
    #把outputs中最后的那一个词(最近预测的那一个词)(outputs[-1])当做下一时刻的输入,reshape((1,1))批量大小和时间步长都是1
    get_input=lambda: torch.tensor([outputs[-1]],device=device).reshape((1,1))
    #处理前缀prefix，把信息存进state
    for y in prefix[1:]:
        #根据已知的prefix，不关心输出，只保留初始隐藏状态
        _,state=net(get_input(),state)
        #outputs接受prefix中字符在词表中的真实对应，没有额外误差
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y,state=net(get_input(),state)
        #y:[1,vocab_size]的向量，下面对y做分类，取出最大的下标位置reshape成为标量，按行分类(dim=1)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    #把整型转换层token，最后用join串联起来
    return ''.join([vocab.idx_to_token[i] for i in outputs])

"""
梯度剪裁:
对于长度为T的序列，我们在迭代中计算T个时间步上的梯度时，将会在反向传播中恒诚成都为O(T)的矩阵乘法链，
当T较大时，可能会产生梯度爆炸或者梯度消失的情况，循环神经网络需要额外的方式来支持稳定的训练
将梯度g通过投影回给定半径(r)的球来截断梯度g
g <-- min(1,r/||g||)
梯度范数永远不会超过r，并且更新后的梯度方向完全与g的原始方向一致
"""
def grad_clipping(net,theta): #@save
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    #把所有层的p的梯度取平方，求和，再求和开根号，把所有层的梯度拉到一起生成一个特别长的向量
    norm=torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

def train_epoch_ch8(net,train_iter,device,loss,updater,use_random_iter):
    state,timer=None,time.time()
    total_loss,total_tokens=0,0
    for X,Y in train_iter:
        if state is None or use_random_iter:
            #如果隐藏状态是None,按照时间步数生成[批量大小，词表大小]的隐藏状态；
            #如果用的是random_iter，由于前后批量不连续，每个批量的初始化隐藏状态必须重新设置
            state=net.begin_state(batch_size=X.shape[0],device=device)
        else:
            """
            sequence_iter由于前后连续，可以将state继承，但是其中梯度要剪裁掉
            detach_()：将张量从计算图中分离出来，切断梯度反向传播的路径，且不再记录梯度。
            state中的做反向传播的时候把先前计算的梯度全部清除，只保留现在和这个批量的梯度计算
            如果不做任何处理：计算图会保留所有历史批量的梯度信息反向传播时,会从当前批量一直传播到第一个批量,导致显存爆炸和训练速度极慢
            """
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()#单个张量(GRU)
            else:
                for s in state:
                    s.detach_()#张量元组(LSTM的(h,c))
        #Y形状: [批量大小, 时间步数] → 转置 → [时间步数, 批量大小] → 展平 → [时间步数*批量大小]
        y=Y.T.reshape(-1)
        X,y=X.to(device),y.to(device)
        #y_hat形状[时间步数*批量大小, 词表大小]
        y_hat,state=net(X,state)
        #y的形状与 y_hat 的形状 [num_steps * batch_size, vocab_size] 匹配，方便计算交叉熵损失
        l=loss(y_hat,y.long()).mean()

        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net,1)
            updater(batch_size=1)
        total_loss+=l.item()*y.numel()
        total_tokens+=y.numel()

    #计算困惑度
    perplexity=math.exp(total_loss/total_tokens)
    speed=total_tokens/(time.time()-timer)
    return perplexity,speed

def train_ch8(net,train_iter,vocab,lr,num_epoches,device,use_random_iter=False):
    loss=nn.CrossEntropyLoss()
    if isinstance(net,nn.Module):
        updater=torch.optim.SGD(net.parameters(),lr=lr)
    else:
        def updater(batch_size):
            with torch.no_grad():
                for param in net.params:
                    param-=lr*param.grad/batch_size
                    param.grad.zero_()
    predict=lambda prefix:predict_ch8(prefix,200,net,vocab,device)
    for epoch in range(num_epoches):
        ppl,speed=train_epoch_ch8(net,train_iter,device,loss,updater,use_random_iter)
        if (epoch+1)%50==0:
            print(predict('time traveller'))
    print(f'困惑度{ppl:.1f},{speed:.1f}词元/秒')
    print(predict('time traveller'))

if __name__ == '__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    batch_size,num_steps=32,35
    num_hiddens=512
    num_epoches,lr=500,1.0
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)

    net=RNNModelScratch(len(vocab),num_hiddens,device,get_params,init_rnn_state,rnn)
    train_ch8(net,train_iter,vocab,lr,num_epoches,device)