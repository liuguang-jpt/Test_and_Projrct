# ============================================================================
# Seq2Seq（序列到序列）机器翻译模型
# 使用 GRU 编码器-解码器架构，将一种语言的句子翻译为另一种语言
# 核心流程：编码器读取源句 → 隐藏状态传递给解码器 → 解码器自回归生成目标句
#
# 核心概念 —— num_steps（时间步数 / 序列长度）：
#   每个句子被截断或填充到固定长度 num_steps，使得一个 batch 内的所有句子
#   等长，可以堆叠成规整的矩形张量。
#      "I love you"           → [I, love, you, <pad>, <pad>]   num_steps=5
#      "Machine learning is"  → [Machine, learning, is, <pad>, <pad>]
#   在训练阶段，num_steps 是固定的超参数（如 10），所有 batch 统一；
#   在推理阶段，num_steps 是最大生成步数，防止解码器无限循环。
#   张量形状中出现的 num_steps，本质上就是"序列方向上有多少个位置"。
# ============================================================================

import collections  # 用于 BLEU 评分中的 n-gram 统计
import time         # 计时训练速度
import math         # 困惑度/BLEU 计算中的 exp、pow
from 机器翻译.机器翻译 import truncate_pad  # 截断/填充函数：将序列裁剪或填充到固定长度
import torch
import torch.nn as nn
from torch import optim  # 导入 torch.optim 模块，后续用 optim.Adam

# 自动选择设备：有 GPU 则用 GPU，否则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 编码器（Encoder）
# 作用：读取源语言句子，通过 GRU 将其编码为一系列隐藏状态
# 输入：(batch_size, num_steps) — 批量源语言词元索引序列
# 输出：
#   - output: (num_steps, batch_size, num_hiddens) — 每个时间步的顶层 GRU 输出
#   - state:  (num_layers, batch_size, num_hiddens) — 各层最终时间步的隐藏状态
# ============================================================================
class Seq2seqEncoder(nn.Module):
    """
    参数说明：
        vocab_size  : 源语言词表大小（输入维度 = one-hot 维度）
        embed_size  : 词嵌入向量的维度（将离散词元映射为稠密向量）
        num_hiddens : GRU 隐藏层维度
        num_layers  : GRU 层数（多层堆叠，层间由 PyTorch 自动连接）
        dropout     : 层间 dropout 比例，防止过拟合（仅当 num_layers > 1 时生效）
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2seqEncoder, self).__init__(**kwargs)
        # 嵌入层：将词元索引（整数）映射为 embed_size 维的可学习稠密向量
        # 权重矩阵形状为 (vocab_size, embed_size)，每行是一个词元的嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU 层：输入 embed_size 维，输出 num_hiddens 维
        # PyTorch 的 GRU 默认 batch_first=False，即输入形状为 (seq_len, batch, input_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X 形状：(batch_size, num_steps) — 每个元素是词元在词表中的索引
        # 经过嵌入层 → (batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # permute(1,0,2)：交换轴 0 和轴 1，将"批量优先"转为"时间步优先"
        # → (num_steps, batch_size, embed_size)，因为 GRU 要求第一维是时间步
        X = X.permute(1, 0, 2)
        # output: (num_steps, batch_size, num_hiddens) — 每个时间步的顶层输出
        # state:  (num_layers, batch_size, num_hiddens) — 最终时间步各层的隐藏状态
        output, state = self.rnn(X)
        return output, state


# ============================================================================
# 解码器（Decoder）
# 作用：接收编码器的最终隐藏状态作为上下文（初始状态），自回归生成目标语言句子
# 输入：(batch_size, num_steps) — 目标语言词元索引序列（训练时用 Teacher Forcing）
# 输出：
#   - output: (batch_size, num_steps, vocab_size) — 每个位置对目标词表中各词元的预测 logits
#   - state:  (num_layers, batch_size, num_hiddens) — 最终隐藏状态
# ============================================================================
class Seq2seqDecoder(nn.Module):
    """
    参数说明：
        vocab_size  : 目标语言词表大小（输出维度）
        embed_size  : 词嵌入向量的维度
        num_hiddens : GRU 隐藏层维度
        num_layers  : GRU 层数
        dropout     : 层间 dropout 比例
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2seqDecoder, self).__init__(**kwargs)
        # 嵌入层：将目标语言词元索引映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU 层：注意输入维度 = embed_size + num_hiddens（因为拼接了上下文向量）
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        # 全连接层（输出层）：将 GRU 的隐藏状态映射到词表空间，生成每个词元的 logits
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        """
        从编码器输出中提取初始隐藏状态
        enc_outputs 是编码器的返回值 (output, state)，
        取 state（即 enc_outputs[1]）作为解码器的初始隐藏状态，
        从而将源语言句子的编码信息传递给解码器。
        """
        return enc_outputs[1]

    def forward(self, X, state):
        # X 形状：(batch_size, num_steps) — 目标语言词元索引
        # 嵌入 → permute → (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # state[-1]：取编码器最后一层（顶层）的最终隐藏状态
        #   形状 (batch_size, num_hiddens) → unsqueeze → (1, batch_size, num_hiddens)
        # repeat：沿时间步维度复制 num_steps 次 → (num_steps, batch_size, num_hiddens)
        # 作用：让解码器的每一个时间步都能"看到"编码器的上下文信息
        # state[-1] 形状为 (batch_size, num_hiddens)，是2维张量。
        # repeat 传入3个参数时，PyTorch 会在左边自动补1维：
        #   (batch_size, num_hiddens) → (1, batch_size, num_hiddens)
        # 然后按 (num_steps, 1, 1) 复制：
        #   第1个参数 num_steps → 沿时间步维复制 num_steps 次
        #   第2个参数 1         → batch_size 维不复制
        #   第3个参数 1         → num_hiddens 维不复制
        # 结果：(num_steps, batch_size, num_hiddens)
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 在特征维度（dim=2）上拼接嵌入向量和上下文向量
        # X 形状 (num_steps, batch_size, embed_size)
        # context 形状 (num_steps, batch_size, num_hiddens)
        # 拼接后 → (num_steps, batch_size, embed_size + num_hiddens)
        X_and_context = torch.cat((X, context), dim=2)
        # GRU 前向传播（初始隐藏状态为编码器传来的 state）
        # output: (num_steps, batch_size, num_hiddens)
        # state:  (num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X_and_context, state)
        # 全连接层将每个时间步的输出映射为词表维度的 logits
        # permute(1,0,2) 转回"批量优先" → (batch_size, num_steps, vocab_size)
        output = self.dense(output).permute(1, 0, 2)
        return output, state

# ============================================================================
# 序列掩码函数
# 作用：将填充位置（超出有效长度的部分）的值替换为指定值（默认为 0）
# 这样在计算损失时，填充位置的贡献被置零，不会影响训练
# ============================================================================
# 序列掩码函数 —— 把 <pad> 位置的贡献清零
# ============================================================================
# 背景：NLP 中句子长度不一，需要填充到统一长度才能组成 batch。
#       但 <pad> 是假词，不应参与损失计算。本函数用一个布尔广播
#       比较来标记有效/填充位置，然后把填充位置的值置为 value。
#
# 参数：
#   X         : (batch_size, maxlen, ...)
#   valid_len : (batch_size,)  每个样本的有效长度
#   value     : 填充位置被赋予的值，默认 0
#
# 示例（batch_size=2, maxlen=8, valid_len=[3,8]）：
#   arange       → [0,1,2,3,4,5,6,7]         形状 (8,)
#   [None, :] 升维成行向量，[:, None] 升维成列向量
#   [None,:]     → [[0,1,2,3,4,5,6,7]]       形状 (1,8)
#   [:,None]     → [[3],[8]]                  形状 (2,1)
#   广播比较得到 mask:
#     row0(len=3): [T, T, T, F, F, F, F, F]   ← 位置3~7是<pad>
#     row1(len=8): [T, T, T, T, T, T, T, T]   ← 全部有效
#   ~mask 取反 → 填充位置赋 value → 完成清零
# ============================================================================
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    # 核心：广播比较生成布尔掩码
    # mask[i,j] = True  ⇔ 第i个样本的第j个位置是真实词（非填充）
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value    # 填充位置置为 value（损失函数中置0即忽略）
    return X


# ============================================================================
# 带掩码的 Softmax 交叉熵损失
# 继承自 nn.CrossEntropyLoss，额外处理填充位置：
#   填充位置的损失权重为 0，不计入最终损失
#   只在有效词元上计算交叉熵
# ============================================================================
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    pred      : 模型预测的 logits，形状 (batch_size, num_steps, vocab_size)
    label     : 真实标签（词元索引），形状 (batch_size, num_steps)
    valid_len : 每个样本的有效长度，形状 (batch_size,)
    """

    def forward(self, pred, label, valid_len):
        # 创建与 label 同形状的全 1 权重张量
        weights = torch.ones_like(label)
        # 将填充位置（超出有效长度的部分）的权重置为 0
        # 这样填充位置的损失贡献为 0
        weights = sequence_mask(weights, valid_len)
        # reduction='none'：不对每个样本的损失做求和/平均，保留每个位置的独立损失值
        self.reduction = 'none'
        # 调用父类 CrossEntropyLoss 的 forward
        # pred.permute(0,2,1)：PyTorch 的 CrossEntropyLoss 要求输入形状为
        #   (batch_size, num_classes, seq_len)，即类别维度在第 2 维
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 将填充位置的损失值清零，然后沿序列维取平均
        #
        # unweighted_loss 形状：(batch_size, num_steps)
        #   每一行是一个样本，每一列是该样本某个时间步的交叉熵损失
        # weights 同形状，填充位置已置 0
        # 相乘后填充位置损失→0，有效位置损失保留
        #
        # mean(dim=1)：沿序列方向（列方向）对每行取平均
        #   样本0: (0.2+0.5+0.3+0+0+0+0+0) / 8 = 0.125
        #   样本1: (0.1+0.4+0.2+0.6+0.3+0.5+0.7+0.2) / 8 = 0.375
        #   (batch_size, num_steps) → (batch_size,)  每行坍缩为一个标量
        #   结果：每个样本得到一个平均损失值
        weights_loss = (unweighted_loss * weights).mean(dim=1)
        return weights_loss


# ============================================================================
# 训练函数
# 使用 Teacher Forcing（教师强制）策略：
#   训练时，解码器每个时间步的输入是目标序列中的"真实"前一个词元，
#   而非模型上一步预测的词元。这样可以加速收敛、避免误差累积。
# ============================================================================
def train_seq2seq(net, data_iter, lr, num_epoches, tgt_vocab, device):
    """
    参数：
        net         : Seq2Seq 模型（包含 encoder 和 decoder）
        data_iter   : 训练数据迭代器，每个 batch 返回 (X, X_valid_len, Y, Y_valid_len)
        lr          : 学习率
        num_epoches : 训练轮数
        tgt_vocab   : 目标语言词表（用于获取 <bos> 特殊标记等）
        device      : 计算设备
    """

    # Xavier 权重初始化函数
    # Xavier 初始化：根据输入/输出维度缩放初始权重范围，
    # 使梯度在深层网络中的方差保持稳定，缓解梯度消失/爆炸
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            # 对全连接层权重做 Xavier 均匀初始化
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            # 对 GRU 的各组权重参数做 Xavier 均匀初始化
            # _flat_weights 包含 GRU 内部所有可学习参数（权重和偏置）
            for param in m._flat_weights:
                if "weights" in param:
                    nn.init.xavier_uniform_(param)

    # 递归地对 net 的所有子模块应用 xavier_init_weights
    net.apply(xavier_init_weights)
    net.to(device)
    # 使用 Adam 优化器（自适应学习率，收敛更快更稳定）
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 带掩码的交叉熵损失：自动忽略填充位置
    loss = MaskedSoftmaxCELoss()
    # 切换到训练模式（启用 dropout 等）
    net.train()

    for epoch in range(num_epoches):
        timer = time.time()  # 记录本轮开始时间，用于计算训练速度
        total_loss, total_num_tokens = 0, 0  # 累计损失和累计有效词元数

        # ================================================================
        # data_iter 的构建过程（在 load_data_nmt 中完成）：
        # ================================================================
        # ① 读取 fra.txt → 每行 "src\ttgt" → 分词 → source[], target[]
        #    例: source[0]=["go","."], target[0]=["va","!"]
        #
        # ② 构建词表，加入特殊标记：<pad>=填充, <bos>=句首, <eos>=句尾
        #
        # ③ token → 索引，并追加 <eos>：
        #    ["go","."] → [12,5,<eos>] → truncate_pad → [12,5,<eos>,<pad>,<pad>]
        #
        # ④ build_array_nmt 为所有句子生成：
        #    src_array     (N, num_steps)  源句 padded 张量
        #    src_valid_len (N,)            每句有效长度（不含<pad>）
        #    tgt_array     (N, num_steps)  目标句 padded 张量
        #    tgt_valid_len (N,)            每句有效长度
        #
        # ⑤ load_array → TensorDataset + DataLoader(batch_size, shuffle=True)
        #    data_iter 每次迭代 yield 一个 batch：
        #    (X[batch], X_valid_len[batch], Y[batch], Y_valid_len[batch])
        #    每个张量形状：
        #      X,Y            → (batch_size, num_steps)
        #      X/Y_valid_len  → (batch_size,)
        # ================================================================
        for batch in data_iter:
            optimizer.zero_grad()  # 清空上一轮的梯度
            # 解包 batch 并全部移到指定设备（GPU/CPU）
            # X: 源语言序列, X_valid_len: 源序列有效长度
            # Y: 目标语言序列, Y_valid_len: 目标序列有效长度
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # 构造解码器输入：在每个样本前添加 <bos>（句子起始标记）
            # Y.shape[0]：当前批量大小
            # [tgt_vocab['<bos>']] * Y.shape[0] → 1维张量，形状 (batch_size,)
            # reshape(-1,1)：-1自动推断为batch_size → 变成列向量 (batch_size, 1)
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            # Y[:,:-1]：去掉每个目标序列的最后一个词元（因为不需要用最后一个词元作为输入）
            # Teacher Forcing：dec_input 的第 t 个位置对应 Y 的第 t 个真实词元
            # 即用 Y[0] 预测 Y[1]，用 Y[1] 预测 Y[2]，...，所以输入比标签左移一位
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)

            # ================================================================
            # 前向传播 —— 完整调用链路
            # ================================================================
            # net 是 EncoderDecoder(Seq2seqEncoder, Seq2seqDecoder) 实例
            # 调用 net(X, dec_input, X_valid_len) 触发 EncoderDecoder.forward:
            #
            # ① enc_outputs = encoder(X, X_valid_len)
            #    X (batch_size, num_steps) → Embedding → permute → GRU
            #    → enc_output (num_steps, batch_size, num_hiddens)
            #    → enc_state  (num_layers, batch_size, num_hiddens)  ← 源句的压缩表示
            #
            # ② dec_state = decoder.init_state(enc_outputs, X_valid_len)
            #    = enc_outputs[1] = enc_state
            #    把编码器的最终隐藏状态"传递"给解码器作为初始状态
            #
            # ③ output, state = decoder(dec_input, dec_state)
            #    a. dec_input → Embedding → permute → (num_steps, batch, embed)
            #    b. enc_state[-1] → repeat → context (num_steps, batch, hiddens)
            #    c. torch.cat((embed, context), dim=2) → 每个时间步都"看到"源句上下文
            #    d. GRU(拼接结果, enc_state) → 解码
            #    e. Dense → permute → (batch_size, num_steps, vocab_size) logits
            #
            # Y_hat: 每个位置对目标词表的预测分数（取 argmax 即得到预测词元）
            # _    : 解码器最终状态（此处不使用，丢弃）
            # ================================================================
            Y_hat, _ = net(X, dec_input, X_valid_len)

            # 计算带掩码的损失：填充位置的损失被自动忽略
            l = loss(Y_hat, Y, Y_valid_len)
            # sum()：将所有样本的损失求和后再反向传播
            l.sum().backward()
            # 统计本批次有效词元总数（用于困惑度计算）
            num_tokens = Y_valid_len.sum()
            # 更新模型参数
            optimizer.step()

            # 在 torch.no_grad() 下累加统计信息，避免构建不必要的计算图
            with torch.no_grad():
                total_loss += l.sum().item()
                total_num_tokens += Y_valid_len.sum()

        # 每 10 轮打印一次训练状态
        if (epoch + 1) % 10 == 0:
            print(f'loss {total_loss:.4f}\t tokens {total_num_tokens:.4f} time {time.time() - timer:.2f}')


# ============================================================================
# 预测（推理）函数
# 使用自回归（Autoregressive）生成：
#   从 <bos> 开始，每一步取预测概率最大的词元（贪心搜索），
#   将其作为下一步的输入，直到遇到 <eos> 或达到最大步数。
# ============================================================================
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device,
                    save_attention_weights=False):
    """
    参数：
        net                     : 训练好的 Seq2Seq 模型
        src_sentence            : 源语言句子（字符串）
        src_vocab, tgt_vocab    : 源语言/目标语言词表
        num_steps               : 最大生成步数（防止无限循环）
        device                  : 计算设备
        save_attention_weights  : 是否保存注意力权重（用于可视化）
    返回：
        译文句子（字符串）和注意力权重列表
    """
    # 切换到评估模式（关闭 dropout 等）
    net.eval()

    # ================================================================
    # ① 源句预处理：字符串 → token索引列表 → padded张量
    # ================================================================
    # src_sentence.lower().split(' ') → token字符串列表，如 ["go", "."]
    # src_vocab(...)                  → 索引列表，如 [12, 5]
    # + [src_vocab['<eos>']]          → 追加<eos>，如 [12, 5, <eos>]
    src_tokens = src_vocab(src_sentence.lower().split(' ')) + [src_vocab['<eos>']]
    # enc_valid_len: (1,)  有效长度=1（单句推理，batch_size=1）
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # truncate_pad → 填充/截断到 num_steps
    # 输入: list长度=len(src_tokens) → 输出: list长度=num_steps
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # ================================================================
    # ② 编码：源句 → 编码器 → 隐藏状态
    # ================================================================
    # torch.tensor(src_tokens) → (num_steps,)
    # unsqueeze(dim=0)          → (1, num_steps)   batch_size=1
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # encoder(enc_X, enc_valid_len):
    #   enc_X         (1, num_steps)         单句源语言索引
    #   enc_valid_len (1,)                  有效长度
    # → enc_outputs = (enc_output, enc_state)
    #     enc_output: (num_steps, 1, num_hiddens)  各时间步顶层输出
    #     enc_state:  (num_layers, 1, num_hiddens) 最终隐藏状态
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # init_state 提取 enc_outputs[1] = enc_state
    # → dec_state: (num_layers, 1, num_hiddens)  解码器初始隐藏状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # ================================================================
    # ③ 解码器初始输入：<bos> 标记
    # ================================================================
    # [tgt_vocab['<bos>']] → 列表 [<bos>索引]
    # torch.tensor(...)    → (1,)  标量张量
    # unsqueeze(dim=0)     → (1, 1)  (batch_size=1, seq_len=1)  当前只有<bos>一个词元
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weights_seq = [], []

    # ================================================================
    # ④ 自回归生成循环：每次生成一个词元，最多 num_steps 步
    # ================================================================
    for _ in range(num_steps):
        # --- 解码器前向传播 ---
        # dec_X:     (1, 1)                       当前输入（初始为<bos>，之后为上一步预测词元）
        # dec_state: (num_layers, 1, num_hiddens)  当前隐藏状态
        # → Y:        (1, 1, vocab_size)           每个位置对词表的预测 logits
        # → dec_state: (num_layers, 1, num_hiddens) 更新后的隐藏状态
        Y, dec_state = net.decoder(dec_X, dec_state)

        # --- 贪心解码：取概率最大的词元 ---
        # Y.argmax(dim=2): (1, 1, vocab_size) → (1, 1)  沿词表维取argmax
        dec_X = Y.argmax(dim=2)

        # --- 提取预测词元索引 ---
        # dec_X.squeeze(dim=0): (1,1) → (1,)  去掉batch维
        # .type(torch.int32).item(): 张量 → Python int
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        if save_attention_weights:
            attention_weights_seq.append(net.decoder.attention_weights)

        # 遇到 <eos> 则停止生成
        if pred == tgt_vocab['<eos>']:
            break

        # 将预测词元加入输出序列
        output_seq.append(pred)

    # ================================================================
    # ⑤ 索引序列 → 自然语言句子
    # ================================================================
    # output_seq: list of int, 如 [23, 7, ...]
    # tgt_vocab.to_tokens(...) → list of str, 如 ["va", "!", ...]
    # ' '.join(...) → "va !"
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weights_seq


# ============================================================================
# BLEU（Bilingual Evaluation Understudy）—— 机器翻译自动评估指标
# ============================================================================
# 动机：人工评估翻译质量太慢太贵。BLEU 用 n-gram 重合度自动打分，
#       分数范围 [0, 1]，越高表示预测译文与参考译文越接近。
#
# ---- 什么是 n-gram？ ----
# n-gram = 连续 n 个词元组成的滑动窗口片段。
# 例: "the cat is on the mat"
#   1-gram: "the", "cat", "is", "on", "the", "mat"                    → 6个
#   2-gram: "the cat", "cat is", "is on", "on the", "the mat"         → 5个
#   3-gram: "the cat is", "cat is on", "is on the", "on the mat"      → 4个
#   4-gram: "the cat is on", "cat is on the", "is on the mat"         → 3个
# 规律: 长度为L的句子，n-gram 数量 = L - n + 1
# 不同阶数衡量不同粒度的翻译质量：
#   1-gram → 词级准确度（"用对了词吗？"）
#   2-gram → 短语流畅度（"词搭配合适吗？"）
#   3-gram → 局部语法   （"部分语序对吗？"）
#   4-gram → 短句结构   （"连续4个词的语序对吗？"）
#
# ---- 核心思想：修正的 n-gram 精度 (Modified Precision) ----
# 普通精度 = 匹配数 / 预测数，但会鼓励模型"疯狂输出高频词"。
# 例：参考="the cat sat"，预测="the the the the the"
#     普通 1-gram 精度 = 5/5 = 1.0  ← 满分但垃圾译文！
#     BLEU 做法：每个参考 n-gram 只允许被匹配一次（截断计数 clipping）。
#       "the" 在参考中出现 1 次，所以预测中最多匹配 1 次。
#       → 修正精度 = 1/5 = 0.2  ← 合理打压
#
# ---- 短句惩罚 (Brevity Penalty) ----
# 短句精度天然高（分母小），BLEU 用 BP 惩罚过短译文：
#   BP = exp(1 - len_ref / len_pred)  当 len_pred < len_ref
#   BP = 1                            当 len_pred >= len_ref
#   例：参考 10 词，预测 3 词 → BP = exp(1-10/3) = exp(-2.33) ≈ 0.097
#
# ---- 多阶几何平均 + 递减权重 ----
# 几何平均的特点：某一阶精度为 0 → 整体 BLEU = 0（一票否决）。
# 权重 0.5^n 让高阶 n-gram 递减贡献（越高阶越难匹配，不应与 1-gram 等权）。
#
# ---- 完整公式 ----
# BLEU = BP × Π_{n=1..k} (p_n)^{0.5^n}
#   其中 p_n = clipped_matches_n / total_pred_ngrams_n
#
# ---- 数值示例 (k=4) ----
# 参考: "the cat is on the mat"
# 预测: "the cat is on the mat"  (完美匹配)
#   p_1=6/6, p_2=5/5, p_3=4/4, p_4=3/3  → p_n 全为 1
#   BP = exp(0) = 1（等长）
#   BLEU = 1 × 1^{0.5} × 1^{0.25} × 1^{0.125} × 1^{0.0625} = 1.0 ✓
# ============================================================================
def bleu(pred_seq, label_seq, k):
    """
    参数：
        pred_seq : 模型生成的预测句子（字符串，词元间用空格分隔）
        label_seq: 参考译文（字符串，词元间用空格分隔）
        k        : 最高 n-gram 阶数（通常 k=4）
    返回：
        BLEU 分数，范围 [0, 1]，越高越好
    """
    # 将句子按空格切分为词元列表
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    # 【已修复】原代码: len_label = len(label_seq)  (Bug: 统计的是字符串长度而非词元数)
    len_pred, len_label = len(pred_tokens), len(label_tokens)

    # ---- 短句惩罚 (Brevity Penalty) ----
    # len_pred < len_label → 1 - len_label/len_pred < 0 → exp(负数) < 1  施加惩罚
    # len_pred >= len_label → min(0, 正数) = 0 → exp(0) = 1           不惩罚
    # 例: 预测 5 词, 参考 10 词 → BP = exp(1-10/5) = exp(-1) ≈ 0.368
    score = math.exp(min(0, 1 - len_label / len_pred))

    # ---- 逐阶计算修正 n-gram 精度 ----
    for n in range(1, k + 1):
        num_matches = 0
        # 统计参考译文中每个 n-gram 的可用次数
        # 用 defaultdict(int) 实现截断计数：每个参考 n-gram 只能被匹配有限次
        # 【已修复】原代码: collections.defaultdict(label_seq)
        #   (Bug: defaultdict 的第一个参数应该是工厂函数如 int，而非字符串)
        label_subs = collections.defaultdict(int)

        # ① 统计参考译文 n-gram 频率（作为匹配配额）
        #   例: 参考 "the cat the" → 2-gram: {"the cat":1, "cat the":1}
        # 【已修复】原代码中此处错误使用了 pred_tokens 构建字典
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i:i + n])] += 1

        # ② 遍历预测译文的每个 n-gram，消耗配额进行匹配
        #   ngram 在参考中存在且配额>0 → 匹配成功，配额-1（防止重复匹配同一参考 n-gram）
        #   ngram 在参考中不存在或配额=0 → 不匹配
        for i in range(len_pred - n + 1):
            ngram = ' '.join(pred_tokens[i:i + n])
            if label_subs[ngram] > 0:
                num_matches += 1
                label_subs[ngram] -= 1       # 消耗一次配额

        # ③ 修正精度 p_n = 匹配数 / 预测句中的 n-gram 总数
        # ④ 权重: 0.5^n（高阶贡献递减）
        #   1-gram: 0.5^1=0.5 | 2-gram: 0.5^2=0.25 | 3-gram: 0.125 | 4-gram: 0.0625
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))

    return score


# ============================================================================
# 【补充说明】词嵌入层（nn.Embedding） vs One-Hot 编码
# ============================================================================
#
# 两者都是将离散词元（token）转换为数值向量的方法，但思路和效果完全不同。
#
# --------------------------------------------------------------------------
# 一、One-Hot 编码（如 RNN.py 中的做法）
# --------------------------------------------------------------------------
#   X = nn.functional.one_hot(X.T, self.vocab_size).type(torch.float32)
#
# 原理：每个词用一个长度为 vocab_size 的向量表示，只有该词索引对应位置为 1，其余全为 0。
#
# 例如，词表 ['我','爱','你','猫','狗']，vocab_size=5：
#   '我' → [1, 0, 0, 0, 0]
#   '猫' → [0, 0, 0, 1, 0]
#   '狗' → [0, 0, 0, 0, 1]
#
# 缺点：
#   1. 维度灾难：词表多大向量就多长（通常几万~几十万维），极其稀疏
#   2. 无语义信息：任意两个词之间的欧氏距离都相等（都是√2），
#      '猫'和'狗'的距离 = '猫'和'电脑'的距离，完全无法捕捉语义关系
#   3. 内存浪费：99.9% 的元素都是 0
#
# --------------------------------------------------------------------------
# 二、词嵌入层（nn.Embedding）—— 本文件中使用的方式
# --------------------------------------------------------------------------
#   self.embedding = nn.Embedding(vocab_size, embed_size)
#
# 原理：维护一张可学习的词嵌入矩阵 W，大小为 (vocab_size × embed_size)，
# 每一行是一个词的稠密向量。输入词元索引 → 查表 → 取出对应行。
#
#                   词嵌入矩阵 W (vocab_size × embed_size)
#                   ┌──────────────────────────────┐
#    输入索引 3 →   │ row 0: [ ... ]              │
#                   │ row 1: [ ... ]              │
#                   │ row 2: [ ... ]              │
#                   │ row 3: [0.2, -0.5, 0.8, 0.1]│ ← 取出这一行
#                   │ row 4: [ ... ]              │
#                   └──────────────────────────────┘
#                   输出: (embed_size,) 的稠密向量
#
# 例如 embed_size=4：
#   '猫'   → [ 0.32, -0.17,  0.85,  0.04 ]
#   '狗'   → [ 0.28, -0.21,  0.79, -0.02 ]   ← 和"猫"向量方向接近 ✓
#   '电脑' → [-0.61,  0.55, -0.12,  0.73 ]   ← 和"猫"向量方向差异大 ✓
#
# --------------------------------------------------------------------------
# 三、核心区别对比
# --------------------------------------------------------------------------
#   ┌──────────┬─────────────────────┬──────────────────────────┐
#   │          │ One-Hot             │ 词嵌入（Embedding）       │
#   ├──────────┼─────────────────────┼──────────────────────────┤
#   │ 维度     │ vocab_size（几万维） │ 可自定义（通常128~1024） │
#   │ 稀疏性   │ 稀疏（只有一个1）    │ 稠密（所有维度都有意义） │
#   │ 语义关系 │ 无，所有词等距       │ 有，语义相近词向量也相近 │
#   │ 参数来源 │ 无参数（固定规则）   │ 可学习参数（随训练优化） │
#   │ 计算方式 │ 等价于矩阵乘法       │ 直接查表 O(1)            │
#   │ 泛化能力 │ 无法泛化            │ 相似语境词获得相似向量   │
#   └──────────┴─────────────────────┴──────────────────────────┘
#
# --------------------------------------------------------------------------
# 四、等价视角
# --------------------------------------------------------------------------
# One-Hot × 嵌入矩阵 = 查表：
#   [0, 0, 1, 0, 0] × W(vocab_size × embed_size) = W 的第 3 行
#
# nn.Embedding 本质上就是用 One-Hot 做矩阵乘法的"高效等价实现"——
# 直接查表，避免构建巨大的稀疏 One-Hot 矩阵。
#
# --------------------------------------------------------------------------
# 五、直观类比：分布式表示
# --------------------------------------------------------------------------
# 把词嵌入想象成给每个词标注了 N 个"属性分数"。假设 embed_size=3，
# 三个维度恰好学到了（这些不是人工指定的，是模型自动学的）：
#
#   维度1 → "生物性"得分
#   维度2 → "大小"得分
#   维度3 → "情感色彩"得分
#
#   '猫'  → [ 0.9, -0.3,  0.5 ]   ← 高生物性，小型，偏正面
#   '狗'  → [ 0.8, -0.1,  0.7 ]   ← 和猫的向量很接近 ✓
#   '岩石'→ [-0.6,  0.2, -0.1 ]   ← 低生物性，和猫差异大 ✓
#   '爱'  → [ 0.1,  0.0,  0.95]   ← 高情感色彩
#
# 核心原理：分布式假设（Distributional Hypothesis）
# "You shall know a word by the company it keeps."
# 出现在相似上下文中的词，自然会获得相似的向量。
