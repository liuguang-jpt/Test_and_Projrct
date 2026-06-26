import os
import sys
import torch

# 解决 Windows 终端 GBK 编码无法打印法语字符的问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------- 导入 Vocab 类 ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from The_Time_Machine.The_Time_Machine import Vocab


def read_data_nmt():
    """读取英法翻译数据集 fra.txt，返回原始文本。

    文件格式：每行一条平行语料，英文和法文用 \\t 分隔。
    示例行：Go.\\tVa !
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "fra.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def preprocess_nmt(text):
    """文本预处理：统一小写 + 标点前加空格。

    例: "Go." → "go ."   (小写化 + '.' 前补空格)
        "Va !" → "va !"  (不间断空格替换为普通空格)
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 替换不间断空格，统一小写
    text = text.replace(' ', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """按行分词，返回二维列表 source[N][tokens] 和 target[N][tokens]。

    例: 输入行 "go .\\tva !" → source[0]=["go","."]  target[0]=["va","!"]
    两个列表一一对齐，长度均为 N（句子数），每个元素是一个 token 列表。
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """截断或填充序列到固定长度 num_steps。

    例: num_steps=5, padding_token='<pad>'
      ["go",".","<eos>"]               → ["go",".","<eos>","<pad>","<pad>"]   (填充)
      ["this","is","too","long","!","?"] → ["this","is","too","long","!"]     (截断)
    """
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将 token 列表转为 padded 张量 + 有效长度。

    输入: lines = [["go","."], ["va","!"], ...]   N个句子
    步骤:
      ① vocab[l]           → token→索引，如 [[12,5], [23,7], ...]
      ② 追加 <eos>          → [[12,5,<eos>], [23,7,<eos>], ...]
      ③ truncate_pad        → 每个句子都变成 num_steps 长度（填充<pad>或截断）
      ④ torch.tensor(...)   → array 形状 (N, num_steps)
      ⑤ (array != <pad>)    → 逐位置比较，True=有效, False=填充
         .sum(1)             → 沿 dim=1 求和，得到每个句子的有效词元数
                              valid_len 形状 (N,)

    返回: array (N, num_steps), valid_len (N,)
    """
    lines = [vocab[l] for l in lines]                # ① token → 索引
    lines = [l + [vocab['<eos>']] for l in lines]     # ② 追加 <eos>
    array = torch.tensor([truncate_pad(               # ③④ 填充+转张量
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # ⑤
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """将多个张量打包为 DataLoader，每次迭代返回一个 batch。

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    每个张量的第 0 维均为 N（样本总数），TensorDataset 将它们按样本对齐。
    DataLoader 每次 yield 一个小批量：
      (X[batch], X_valid_len[batch], Y[batch], Y_valid_len[batch])
    训练时 shuffle=True 打乱顺序，防止模型记忆样本顺序。
    """
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """完整的机器翻译数据加载流水线（对外统一入口）。

    调用链:
      read_data_nmt   → 读取 fra.txt 原始文本
      preprocess_nmt   → 小写 + 标点分离 ("Go." → "go .")
      tokenize_nmt     → 按行分词 → source[N][tokens], target[N][tokens]
      Vocab(...)       → 构建源/目标词表（含 <pad>/<bos>/<eos>）
      build_array_nmt  → token→索引+<eos>+pad → array (N,num_steps) + valid_len (N,)
      load_array       → TensorDataset + DataLoader → data_iter

    返回:
      data_iter  — 每次迭代 yield (X, X_len, Y, Y_len)，各形状见 load_array
      src_vocab  — 源语言词表（支持 token↔index 互转）
      tgt_vocab  — 目标语言词表
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_word=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_word=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# ====================== Seq2Seq 模型定义 ======================
import torch.nn as nn
import math
import time
import matplotlib.pyplot as plt


class Seq2SeqEncoder(nn.Module):
    """编码器：Embedding → GRU"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X: (batch_size, num_steps)
        X = self.embedding(X).permute(1, 0, 2)   # (num_steps, batch, embed)
        output, state = self.rnn(X)               # state: (num_layers, batch, hiddens)
        return output, state


class Seq2SeqDecoder(nn.Module):
    """解码器：Embedding → GRU(拼接编码器上下文) → Linear"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]   # 取编码器的最终隐状态

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)                       # (num_steps, batch, embed)
        context = state[-1].repeat(X.shape[0], 1, 1)                 # 上下文：最后层隐状态
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)                 # (batch, num_steps, vocab)
        return output, state


class EncoderDecoder(nn.Module):
    """编码器-解码器架构"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)


# ====================== 训练 ======================
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    loss_history, ppl_history = [], []

    for epoch in range(num_epochs):
        timer = time.time()
        total_loss = total_tokens = 0

        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)  # 教师强制

            optimizer.zero_grad()
            Y_hat, _ = net(X, dec_input)
            loss = loss_fn(Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            num_tokens = Y_valid_len.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        train_loss = total_loss / (total_tokens + 1e-8)
        ppl = math.exp(train_loss)
        loss_history.append(train_loss)
        ppl_history.append(ppl)

        if (epoch + 1) % 50 == 0:
            speed = total_tokens / (time.time() - timer)
            print(f'[Epoch {epoch + 1:3d}] loss={train_loss:.3f}, ppl={ppl:.1f}, '
                  f'speed={speed:.0f} tok/s')
            show_examples(net, src_vocab, tgt_vocab, num_steps=10, device=device)

    print(f'\n训练完成  final loss={train_loss:.3f}  final ppl={ppl:.1f}')
    return loss_history, ppl_history


# ====================== 预测 / 翻译 ======================
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.tensor(src_tokens, device=device).unsqueeze(0)

    enc_outputs = net.encoder(enc_X)
    dec_state = net.decoder.init_state(enc_outputs)

    dec_X = torch.tensor([tgt_vocab['<bos>']], device=device).unsqueeze(0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(0).item()
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)

    net.train()
    return ' '.join(tgt_vocab.to_token(output_seq))


def show_examples(net, src_vocab, tgt_vocab, num_steps, device, n=3):
    test_sentences = ['go .', 'wow !', 'help !', 'stop !', 'wait !', 'i lost .']
    print('  ' + '-' * 50)
    for sent in test_sentences[:n]:
        translation = predict_seq2seq(net, sent, src_vocab, tgt_vocab, num_steps, device)
        print(f'  {sent:<20} →  {translation}')
    print('  ' + '-' * 50)


# ====================== 可视化 ======================
def plot_training(loss_history, ppl_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Seq2Seq 机器翻译 — 训练曲线', fontsize=13, fontweight='bold')

    ax1.plot(loss_history, 'b-', linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss (final: {loss_history[-1]:.3f})')
    ax1.grid(True, alpha=0.3)

    ax2.plot(ppl_history, 'r-', linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title(f'Perplexity (final: {ppl_history[-1]:.1f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'seq2seq_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'训练曲线已保存至: {save_path}')
    plt.show()


# ====================== 主程序 ======================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # ---------- 超参数 ----------
    embed_size = 32
    num_hiddens = 32
    num_layers = 2
    dropout = 0.1
    batch_size = 64
    num_steps = 8
    lr = 0.005
    num_epochs = 300

    # ---------- 加载数据 ----------
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, num_examples=600)
    print(f'源词表: {len(src_vocab)}  |  目标词表: {len(tgt_vocab)}  |  '
          f'批次数: {len(train_iter)}')

    # ---------- 构建模型 ----------
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    print(f'总参数量: {sum(p.numel() for p in net.parameters()):,}')
    print(f'\n{"=" * 55}')
    print('开始训练 Seq2Seq 机器翻译模型')
    print(f'{"=" * 55}\n')

    loss_history, ppl_history = train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # ---------- 最终翻译效果 ----------
    print(f'\n{"=" * 55}')
    print('最终翻译效果展示')
    print(f'{"=" * 55}')
    test_sentences = [
        'go .', 'wow !', 'help !', 'stop !', 'wait !',
        'i lost .', 'i try .', 'cheers !', 'come on .', 'get up .'
    ]
    for sent in test_sentences:
        translation = predict_seq2seq(net, sent, src_vocab, tgt_vocab, num_steps=10, device=device)
        print(f'  {sent:<20} →  {translation}')

    # ---------- 可视化 ----------
    plot_training(loss_history, ppl_history)
