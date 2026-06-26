import torch
from RNN import RNNModelScratch,train_ch8
from The_Time_Machine.The_Time_Machine import load_data_time_machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size,num_steps=32,35
train_loader,vocab=load_data_time_machine(batch_size,num_steps)

def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01

    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device))

    W_fx,W_fh,b_f=three()
    W_Ix,W_Ih,b_I=three()
    W_Ox,W_Oh,b_O=three()
    W_xc,W_hc,b_c=three()
    W_hq,b_q=torch.randn((num_hiddens,num_outputs),device=device)*0.01,torch.zeros(num_outputs,device=device)

    params=[W_fx,W_fh,b_f,W_Ix,W_Ih,b_I,W_xc,W_hc,b_c,W_Ox,W_Oh,b_O,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),
            torch.zeros((batch_size,num_hiddens),device=device))

def LSTM(inputs,state,params):
    W_fx,W_fh,b_f,W_Ix,W_Ih,b_I,W_xc,W_hc,b_c,W_Ox,W_Oh,b_O,W_hq,b_q=params
    H,C=state
    outputs=[]
    for X in inputs:
        F=torch.sigmoid(X@W_fx+H@W_fh+b_f)
        I=torch.sigmoid(X@W_Ix+H@W_Ih+b_I)
        O=torch.sigmoid(X@W_Ox+H@W_Oh+b_O)
        C_tilde=torch.tanh(X@W_xc+H@W_hc+b_c)
        C=F*C+I*C_tilde
        H=O*torch.tanh(C)
        Y=H@W_hq+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,C)

vocab_size,num_hiddens,device=len(vocab),512,device
num_epoches,lr=500,1
model=RNNModelScratch(vocab_size,num_hiddens,device,get_params,init_lstm_state,forward_fn=LSTM)
train_ch8(model,train_loader,vocab,lr,num_epoches,device)

