"""
Seq2Seq 机器翻译 — 带 Attention 开关的独立实现
    ・attention=False → Vanilla Seq2Seq（仅用编码器最终隐状态初始化解码器）
    ・attention=True  → Bahdanau Additive Attention（每步动态加权编码器输出）
"""
import os
import sys
import math
import time
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')


# ======================== 数据加载（自包含） ========================
def read_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '机器翻译', 'fra.txt')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess(text):
    text = text.replace(' ', ' ').replace('\xa0', ' ').lower()
    out = [' ' + c if i > 0 and c in set(',.!?') and text[i - 1] != ' ' else c
           for i, c in enumerate(text)]
    return ''.join(out)


def tokenize(text, num_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i >= num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            src.append(parts[0].split(' '))
            tgt.append(parts[1].split(' '))
    return src, tgt


class Vocab:
    """词表：token ↔ 索引 双向映射"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = collections.Counter(
            t for line in tokens for t in line if len(tokens) > 0 and isinstance(tokens[0], list)
        ) if tokens and isinstance(tokens[0], list) else collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(t) for t in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[i] for i in indices]

    @property
    def unk(self):
        return 0


def truncate_pad(line, num_steps, pad_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [pad_token] * (num_steps - len(line))


def build_array(lines, vocab, num_steps):
    lines = [vocab[l] + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).sum(1)
    return array, valid_len


def load_data(batch_size, num_steps, num_examples=600):
    text = preprocess(read_data())
    src, tgt = tokenize(text, num_examples)
    src_vocab = Vocab(src, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(tgt, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_arr, src_len = build_array(src, src_vocab, num_steps)
    tgt_arr, tgt_len = build_array(tgt, tgt_vocab, num_steps)
    dataset = torch.utils.data.TensorDataset(src_arr, src_len, tgt_arr, tgt_len)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab


# ======================== 注意力机制 ========================
class AdditiveAttention(nn.Module):
    """Bahdanau 加性注意力"""
    def __init__(self, key_dim, query_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries: (batch, query_dim)   — 当前解码器隐状态
        # keys:    (src_len, batch, key_dim) — 编码器所有时间步输出
        # values:  (src_len, batch, val_dim) — 同 keys
        q = self.W_q(queries).unsqueeze(0)         # (1, batch, hidden)
        k = self.W_k(keys)                          # (src_len, batch, hidden)
        features = torch.tanh(q + k)                # (src_len, batch, hidden)
        scores = self.W_v(features).squeeze(-1)     # (src_len, batch)

        if valid_lens is not None:
            mask = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1) >= valid_lens
            scores.masked_fill_(mask, -1e9)

        attn_weights = F.softmax(scores, dim=0)     # (src_len, batch)
        attn_weights = self.dropout(attn_weights)
        context = torch.sum(attn_weights.unsqueeze(-1) * values, dim=0)  # (batch, val_dim)
        return context, attn_weights


# ======================== 编码器 ========================
class Encoder(nn.Module):
    """双向 GRU 编码器"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, X):
        # X: (batch, src_len)
        X = self.embedding(X).permute(1, 0, 2)          # (src_len, batch, embed)
        outputs, state = self.rnn(X)                     # outputs: (src_len, batch, 2*hidden)
        return outputs, state                            # state: (2*layers, batch, hidden)


# ======================== 解码器（Attention 可开关） ========================
class Decoder(nn.Module):
    """单向 GRU 解码器，use_attention 控制是否启用注意力"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 use_attention=True, dropout=0.1):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        enc_out_dim = hidden_dim * 2                            # 编码器双向 → 2*hidden

        if use_attention:
            rnn_in_dim = embed_dim + enc_out_dim                # embedding + 注意力上下文
            self.attention = AdditiveAttention(
                key_dim=enc_out_dim, query_dim=enc_out_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            rnn_in_dim = embed_dim                              # 只用 embedding

        self.rnn = nn.GRU(rnn_in_dim, enc_out_dim, num_layers, dropout=dropout)
        self.dense = nn.Linear(enc_out_dim, vocab_size)

    def _transform_state(self, enc_state):
        """把编码器双向 state (2*layers, batch, hidden) 转为解码器单向 (layers, batch, 2*hidden)"""
        # enc_state: (2*L, batch, H) → (L, batch, 2*H)
        return enc_state.reshape(self.num_layers, 2, -1, self.hidden_dim)\
                        .permute(0, 2, 1, 3)\
                        .reshape(self.num_layers, -1, self.hidden_dim * 2)

    def init_state(self, enc_outputs, enc_valid_lens=None):
        outputs, state = enc_outputs
        if self.use_attention:
            self.enc_outputs = outputs
            self.enc_valid_lens = enc_valid_lens
        return self._transform_state(state)

    def forward(self, X, state):
        # X: (batch, tgt_len)
        X = self.embedding(X).permute(1, 0, 2)              # (tgt_len, batch, embed)
        self.attn_weights_history = []                       # 记录注意力权重（仅 attention 模式）

        outputs = []
        for x in X:                                          # 逐时间步
            x = x.unsqueeze(0)                               # (1, batch, embed)
            if self.use_attention:
                query = state[-1]                            # (batch, 2*hidden)
                context, attn_w = self.attention(
                    query, self.enc_outputs, self.enc_outputs, self.enc_valid_lens)
                self.attn_weights_history.append(attn_w)
                rnn_in = torch.cat((x, context.unsqueeze(0)), dim=-1)
            else:
                rnn_in = x                                   # 无注意力，纯依赖隐状态传递

            out, state = self.rnn(rnn_in, state)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0)                  # (tgt_len, batch, 2*hidden)
        outputs = self.dense(outputs).permute(1, 0, 2)      # (batch, tgt_len, vocab)
        return outputs, state


# ======================== 编码器-解码器 ========================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens=None):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(dec_X, dec_state)


# ======================== 训练 ========================
def train_model(model, data_iter, lr, num_epochs, tgt_vocab, label=''):
    model.to(device)

    # Xavier 初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'], reduction='none')
    history = {'loss': [], 'ppl': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_tokens = 0, 0

        for src, src_len, tgt, tgt_len in data_iter:
            src, src_len, tgt, tgt_len = src.to(device), src_len.to(device), tgt.to(device), tgt_len.to(device)
            bos = torch.full((tgt.shape[0], 1), tgt_vocab['<bos>'], device=device)
            dec_in = torch.cat([bos, tgt[:, :-1]], dim=1)

            optimizer.zero_grad()
            out, _ = model(src, dec_in, src_len)
            loss = loss_fn(out.reshape(-1, out.shape[-1]), tgt.reshape(-1))
            mask = (tgt != tgt_vocab['<pad>']).float().reshape(-1)
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / (total_tokens + 1)
        ppl = math.exp(avg_loss)
        history['loss'].append(avg_loss)
        history['ppl'].append(ppl)

        if (epoch + 1) % 50 == 0:
            print(f'  [{label}] Epoch {epoch+1:3d}  loss={avg_loss:.3f}  ppl={ppl:.1f}')

    return history


# ======================== 预测 ========================
def translate(model, sentence, src_vocab, tgt_vocab, num_steps, device=device):
    """将一句英文翻译为法文"""
    model.eval()
    tokens = src_vocab[sentence.lower().split(' ')] + [src_vocab['<eos>']]
    valid_len = torch.tensor([len(tokens)], device=device)
    tokens = truncate_pad(tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.tensor(tokens, device=device).unsqueeze(0)  # (1, src_len)

    enc_out = model.encoder(enc_X)
    dec_state = model.decoder.init_state(enc_out, valid_len)
    dec_X = torch.tensor([[tgt_vocab['<bos>']]], device=device)

    pred_ids, attn_weights = [], []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(0).item()
        if pred == tgt_vocab['<eos>']:
            break
        pred_ids.append(pred)
        if model.decoder.use_attention and model.decoder.attn_weights_history:
            attn_weights.append(model.decoder.attn_weights_history[-1])

    return ' '.join(tgt_vocab.to_tokens(pred_ids)), attn_weights


# ======================== 可视化 ========================
def plot_comparison(hist_vanilla, hist_attn):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Seq2Seq 机器翻译：Attention 开关对比', fontsize=14, fontweight='bold')
    colors = {'vanilla': '#3498db', 'attention': '#e74c3c'}

    # Loss 曲线
    ax = axes[0]
    ax.plot(hist_vanilla['loss'], color=colors['vanilla'], label=f"无 Attention (final={hist_vanilla['loss'][-1]:.3f})")
    ax.plot(hist_attn['loss'], color=colors['attention'], label=f"有 Attention (final={hist_attn['loss'][-1]:.3f})")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # PPL 曲线
    ax = axes[1]
    ax.plot(hist_vanilla['ppl'], color=colors['vanilla'], label=f"无 Attention (ppl={hist_vanilla['ppl'][-1]:.1f})")
    ax.plot(hist_attn['ppl'], color=colors['attention'], label=f"有 Attention (ppl={hist_attn['ppl'][-1]:.1f})")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Perplexity'); ax.set_title('Perplexity')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 柱状图对比
    ax = axes[2]
    models = ['无 Attention', '有 Attention']
    final_ppl = [hist_vanilla['ppl'][-1], hist_attn['ppl'][-1]]
    bars = ax.bar(models, final_ppl, color=[colors['vanilla'], colors['attention']])
    ax.set_ylabel('Final Perplexity'); ax.set_title('最终困惑度对比')
    for bar, p in zip(bars, final_ppl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{p:.1f}', ha='center', fontweight='bold', fontsize=13)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seq2seq_attention_compare.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n对比图已保存: {save_path}')
    plt.show()


def plot_attention(sentence, translation, attn_weights, src_vocab, tgt_vocab):
    """绘制注意力热力图"""
    if not attn_weights:
        print('（无注意力权重可绘制）')
        return

    src_tokens = sentence.lower().split(' ') + ['<eos>']
    tgt_tokens = translation.split(' ')

    attn_mat = torch.cat([w.detach().cpu() for w in attn_weights], dim=1)
    attn_mat = attn_mat[:len(src_tokens), :len(tgt_tokens)]

    fig, ax = plt.subplots(figsize=(max(6, len(tgt_tokens) * 0.6),
                                    max(4, len(src_tokens) * 0.5)))
    im = ax.imshow(attn_mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(tgt_tokens))); ax.set_xticklabels(tgt_tokens, rotation=45)
    ax.set_yticks(range(len(src_tokens))); ax.set_yticklabels(src_tokens)
    ax.set_xlabel('Decoder (French)'); ax.set_ylabel('Encoder (English)')
    ax.set_title(f'Attention 权重: "{sentence}" → "{translation}"')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attention_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'注意力热力图已保存: {save_path}')
    plt.show()


# ======================== 主程序 ========================
if __name__ == '__main__':
    # ---------- 公共参数 ----------
    BATCH_SIZE, NUM_STEPS = 64, 8
    EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT = 32, 32, 2, 0.1
    LR, NUM_EPOCHS = 0.005, 200

    # ---------- 加载数据 ----------
    data_iter, src_vocab, tgt_vocab = load_data(BATCH_SIZE, NUM_STEPS)
    print(f'数据: {len(data_iter)} batches | 源词表: {len(src_vocab)} | 目标词表: {len(tgt_vocab)}\n')

    # ============= 实验 1: 无 Attention (Vanilla Seq2Seq) =============
    print('='*55)
    print('实验 1: 无 Attention (Vanilla Seq2Seq)')
    print('='*55)
    enc1 = Encoder(len(src_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    dec1 = Decoder(len(tgt_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, use_attention=False, dropout=DROPOUT)
    model1 = Seq2Seq(enc1, dec1)
    print(f'参数量: {sum(p.numel() for p in model1.parameters()):,}')
    hist_vanilla = train_model(model1, data_iter, LR, NUM_EPOCHS, tgt_vocab, label='Vanilla')

    # ============= 实验 2: 有 Attention =============
    print(f'\n{"="*55}')
    print('实验 2: 有 Attention (Bahdanau)')
    print('='*55)
    enc2 = Encoder(len(src_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    dec2 = Decoder(len(tgt_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, use_attention=True, dropout=DROPOUT)
    model2 = Seq2Seq(enc2, dec2)
    print(f'参数量: {sum(p.numel() for p in model2.parameters()):,}')
    hist_attn = train_model(model2, data_iter, LR, NUM_EPOCHS, tgt_vocab, label='Attn')

    # ============= 翻译效果对比 =============
    test_sentences = ['go .', 'wow !', 'help !', 'stop !', 'wait !',
                      'i lost .', 'i try .', 'cheers !', 'come on .', 'get up .']

    print(f'\n{"="*65}')
    print(f'{"英文原文":<25} {"无 Attention":<22} {"有 Attention":<22}')
    print(f'{"="*65}')
    for sent in test_sentences:
        trans1, _ = translate(model1, sent, src_vocab, tgt_vocab, NUM_STEPS)
        trans2, attn2 = translate(model2, sent, src_vocab, tgt_vocab, NUM_STEPS)
        print(f'{sent:<25} {trans1:<22} {trans2:<22}')
    print(f'{"="*65}')

    # ============= 注意力可视化 =============
    chosen_src = 'help !'
    trans_str, attn_weights = translate(model2, chosen_src, src_vocab, tgt_vocab, NUM_STEPS)
    plot_attention(chosen_src, trans_str, attn_weights, src_vocab, tgt_vocab)

    # ============= 训练曲线对比 =============
    plot_comparison(hist_vanilla, hist_attn)
