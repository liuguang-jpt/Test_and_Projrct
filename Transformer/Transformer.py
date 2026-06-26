"""
Transformer 从零实现教程
=======================
按照论文 "Attention Is All You Need" (Vaswani et al., 2017) 的架构，
从零开始逐步实现 Transformer 模型。

每一个组件都有详细的中文注释，解释"是什么"和"为什么"。

目录:
  1.  Scaled Dot-Product Attention — 注意力机制的核心数学
  2.  Multi-Head Attention — 多个注意力头的并行计算
  3.  Position-wise Feed-Forward Network — 位置维度的非线性变换
  4.  Positional Encoding — 让模型知道 token 的位置
  5.  Encoder Layer — 编码器的单层
  6.  Encoder — 堆叠的编码器
  7.  Decoder Layer — 解码器的单层(含 Masked Attention + Cross Attention)
  8.  Decoder — 堆叠的解码器
  9.  Transformer — 完整的 Encoder-Decoder 模型
  10. 训练示例 — 用一个简单的复制任务验证模型

运行方式:
  python Transformer.py          # 训练并测试
  python Transformer.py --test   # 仅测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 第1步: Scaled Dot-Product Attention (缩放点积注意力)
# ==============================================================================
"""
注意力机制的本质是:给定一个查询(query)，去一组键值对(key-value)中检索相关信息。

直觉类比:你去图书馆找书。
  - Query: 你想要什么? ("我想了解深度学习")
  - Key:   每本书的标题/摘要 (用于匹配你的需求)
  - Value: 每本书的完整内容 (匹配后实际返回的信息)

计算过程:
  1. 计算 Q 和每个 K 的相似度 (点积)
  2. 除以 sqrt(d_k) 进行缩放 (防止点积过大导致 softmax 梯度消失)
  3. 用 softmax 归一化成概率分布
  4. 用这个分布对 V 加权求和

公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
"""


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    计算 Scaled Dot-Product Attention。

    参数:
        query:  shape (batch, num_heads, seq_len_q, d_k)
                - batch:      批次大小 (一次处理几条序列)
                - num_heads:  注意力头数 (多头并行的"视角"数)
                - seq_len_q:  查询序列的 token 数量
                - d_k:        每个头的 query/key 向量维度 (= d_model / num_heads)
                语义: "我想找什么" — 当前 token 发出的检索请求
        key:    shape (batch, num_heads, seq_len_k, d_k)
                - batch:      批次大小
                - num_heads:  注意力头数
                - seq_len_k:  被查询序列的 token 数量 (通常 seq_len_k == seq_len_q)
                - d_k:        每个头的向量维度
                语义: "我是什么" — 每个 token 的索引标签, 用于与 query 匹配
        value:  shape (batch, num_heads, seq_len_k, d_v)   (通常 d_v == d_k)
                - batch:      批次大小
                - num_heads:  注意力头数
                - seq_len_k:  被查询序列的 token 数量
                - d_v:        每个头的 value 向量维度
                语义: "我包含什么信息" — query 匹配后实际取出的内容
        mask:   shape (batch, num_heads, seq_len_q, seq_len_k) 或可广播的子形状
                - True/1  表示"可以关注"
                - False/0 表示"屏蔽"(设为 -inf, softmax 后概率为 0)
        dropout: torch.nn.Dropout 模块, 可选

    返回:
        output:            shape (batch, num_heads, seq_len_q, d_v)
                           每个 query 位置融合了所有关注的 value 后的输出
        attention_weights: shape (batch, num_heads, seq_len_q, seq_len_k)
                           每行是一个概率分布: 一个 query 对所有 key 的注意力分配
    """
    d_k = query.size(-1)

    # --- 步骤1: 计算注意力分数 ---
    # Q @ K^T: (batch, heads, seq_q, d_k) x (batch, heads, d_k, seq_k)
    #        = (batch, heads, seq_q, seq_k)
    # 每个 score[i,j] 表示第 i 个 query token 对第 j 个 key token 的"关注程度"
    scores = torch.matmul(query, key.transpose(-2, -1))

    # --- 步骤2: 缩放 ---
    # 为什么除以 sqrt(d_k)?
    # 当 d_k 很大时, 点积的方差会变大 (≈d_k), softmax 的输入值很大时,
    # 梯度趋近于 0 (softmax 进入饱和区), 模型学不动。
    # 除以 sqrt(d_k) 使得方差稳定在 1 左右, 保持梯度健康。
    scores = scores / math.sqrt(d_k)

    # --- 步骤3: 应用掩码 (如果有) ---
    # 两种常见的 mask:
    #   - Padding mask: 把 <pad> token 的位置设为 -inf, 这样 softmax 后概率为 0
    #   - Causal mask (look-ahead mask): 在解码器中, 防止看到未来的 token
    if mask is not None:
        # mask 中 True/1 的位置会被设为 -inf
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # --- 步骤4: Softmax 得到注意力权重 ---
    # 每行是一个概率分布: 一个 query 对所有 key 的注意力分配
    attention_weights = F.softmax(scores, dim=-1)

    # --- 步骤5: Dropout (训练时) ---
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # --- 步骤6: 加权求和 ---
    # (batch, heads, seq_q, seq_k) x (batch, heads, seq_k, d_v)
    # = (batch, heads, seq_q, d_v)
    # 每个 query 位置现在融合了它关注的所有 value 的信息
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# ==============================================================================
# 第2步: Multi-Head Attention (多头注意力)
# ==============================================================================
"""
为什么需要"多头"?
  单头注意力只能学到一种"关注模式"。比如在翻译 "I love you" 时:
  - 头1 可能关注主语-谓语关系 (I → love)
  - 头2 可能关注宾语 (you → love 的宾语)
  - 头3 可能关注位置相邻关系
  - 头4 可能关注长距离依赖

多头 = 用不同的投影矩阵 W_Q, W_K, W_V 把输入投影到不同的"子空间",
在每个子空间分别做注意力, 然后拼接起来。

这就是"ensemble(集成)"思想的体现:多个视角看同一份数据, 信息更丰富。
"""


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块。

    工作流程:
      1. 将输入 X 通过线性层投影成 Q, K, V
      2. 按头数拆分 (分头)
      3. 每个头独立计算 Scaled Dot-Product Attention
      4. 拼接所有头的输出 (合头)
      5. 再过一个线性层 (输出投影)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        参数:
            d_model:   模型的总维度 (论文中是 512)
                       所有层输入输出的统一维度, 也是 Q/K/V 投影的维度
            num_heads: 注意力头的数量 (论文中是 8)
                       将 d_model 均匀拆分为 num_heads 份, 每份独立做注意力
            dropout:   Dropout 比率 (论文中是 0.1)
                       作用于注意力权重上, 随机丢弃一些连接防止过拟合
        """
        super().__init__()
        # 确保 d_model 能被 num_heads 整除
        # 因为我们要把 d_model 均匀分给每个头
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度 (论文中是 64 = 512/8)

        # --- 四个线性投影层 ---
        # W_Q, W_K, W_V: 将输入投影到 d_model 维的 Q, K, V 空间
        # 用 bias=False 因为 LayerNorm 之后不需要偏置
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # W_O: 将拼接后的多头输出投影回 d_model 维
        # 这是多头拼接后的"信息融合层"
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播。

        参数:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)  -- 通常 seq_len_k == seq_len_v
            mask:  可选的注意力掩码

        返回:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # --- 阶段1: 线性投影 + 分头 ---
        # 投影: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # 分头 (split heads):
        # 将 d_model 拆成 num_heads 份, 每份 d_k 维
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        #
        # 为什么要把 num_heads 维度放到前面?
        # 方便后面并行计算: 每个头独立做矩阵乘法时,
        # batch 和 num_heads 都可以当作"批"来处理
        #-1：自动推断维度大小
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # --- 阶段2: 计算注意力 ---
        # 对 mask 做维度扩展, 使其适配多头的 shape
        # 如果传入了 mask: (batch, 1, seq_len_q, seq_len_k) 或 (batch, seq_len_q, seq_len_k)
        if mask is not None:
            # 给 mask 增加 head 维度, 使其可广播到 (batch, num_heads, seq_q, seq_k)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # 调用前面定义的 Scaled Dot-Product Attention
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )

        # --- 阶段3: 合头 (concat heads) ---
        # (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, num_heads, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # -> (batch, seq_len_q, d_model)
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # --- 阶段4: 输出投影 ---
        # 将拼接后的多视角信息融合到统一的表示空间
        output = self.W_O(attn_output)

        # 返回注意力权重用于可视化分析
        return output, attn_weights


# ==============================================================================
# 第3步: Position-wise Feed-Forward Network (位置前馈网络)
# ==============================================================================
"""
Attention 层做的是 token 之间的"信息混合"(不同位置之间的交互),
而 FFN 做的是每个位置内部的"特征变换"(同一个位置的不同维度之间)。

FFN 对每个位置独立地做相同的非线性变换:
  FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

从维度变化看:
  d_model (512) -> d_ff (2048) -> d_model (512)

为什么需要这个?
  Attention 是线性加权 (虽然是"软"选择, 但本质还是加权求和),
  表达能力有限。FFN 引入了非线性 (ReLU/GELU), 让模型能学到更复杂的
  特征变换。可以理解为: Attention 负责"看哪里", FFN 负责"怎么理解看到的东西"。

论文中 d_ff = 2048, 是 d_model (512) 的 4 倍。
这个 4 倍放大-压缩结构给了足够的容量做特征变换。
"""


class PositionwiseFeedForward(nn.Module):
    """
    位置级别的前馈网络: 对每个位置独立应用相同的两层全连接网络。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数:
            d_model: 输入和输出的维度
            d_ff:    中间隐藏层的维度 (通常是 d_model 的 4 倍)
            dropout: Dropout 比率
        """
        super().__init__()
        # 第一层: d_model -> d_ff (放大)
        self.linear1 = nn.Linear(d_model, d_ff)
        # 第二层: d_ff -> d_model (压缩回原维度, 方便残差连接)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            output: (batch, seq_len, d_model)
        """
        # linear1 -> ReLU -> dropout -> linear2
        # 注意: Transformer 论文用的是 ReLU
        # 后来的实现(如 BERT, GPT) 多用 GELU, 效果略好
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ==============================================================================
# 第4步: Positional Encoding (位置编码)
# ==============================================================================
"""
Transformer 没有 RNN 那样的递归结构, 也没有 CNN 那样的滑动窗口。
Self-Attention 本身是"位置无关"的 — 它只看 token 之间的相似度,
不知道 token 在序列中的位置。

所以我们需要显式地告诉模型: "这个词在句子的第几个位置"。

位置编码的方案有很多:
  - 可学习的 (learned): 像 BERT 那样, 随机初始化然后训练
  - 固定的正弦/余弦 (sinusoidal): 论文中的原始方案
  - 相对位置编码 (relative): T5, Transformer-XL 等
  - 旋转位置编码 (RoPE): LLaMA, GPT-NeoX 等现代模型

这里实现论文中的 sinusoidal 方案:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

直觉: 每个位置是一个由不同频率的正弦/余弦波组成的"指纹"。
  - 低频维度 (i 小, 分母小, 波长长): 对远距离位置更敏感 → 捕捉全局位置
  - 高频维度 (i 大, 分母大, 波长短): 对近距离位置更敏感 → 捕捉局部位置
"""


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码。
    产生一个 (1, max_len, d_model) 的矩阵, 加到输入的 embedding 上。
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        参数:
            d_model: 模型的维度
            max_len: 支持的最大序列长度
            dropout: Dropout 比率
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # --- 创建位置编码矩阵 pe: (max_len, d_model) ---
        # 步骤1: 创建位置索引 (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # 步骤2: 计算分母中的频率项
        # div_term = 1 / (10000^(2i/d_model)) = exp(-2i * log(10000) / d_model)
        # 对于每个维度索引 i, 计算对应的缩放因子
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # 步骤3: 计算位置编码
        # pe[:, 0::2] = sin(position * div_term)   -- 偶数维度
        # pe[:, 1::2] = cos(position * div_term)   -- 奇数维度
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # sin 用于偶数位
        pe[:, 1::2] = torch.cos(position * div_term)  # cos 用于奇数位

        # 步骤4: 增加 batch 维度, 注册为 buffer
        # buffer: 不是参数(不会被优化), 但会随着模型保存/加载
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            (batch, seq_len, d_model)  -- 加上位置编码后, 再做 dropout
        """
        seq_len = x.size(1)
        # 取出对应长度的位置编码, 加到输入上
        # self.pe[:, :seq_len] 的形状是 (1, seq_len, d_model)
        # 通过广播加到 batch 中的每个样本
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# ==============================================================================
# 第5步: Encoder Layer (编码器层)
# ==============================================================================
"""
一个 Encoder Layer 包含两个子层:
  1. Multi-Head Self-Attention → 让每个 token 看到序列中所有其他 token
  2. Position-wise FFN         → 对每个 token 的特征做非线性变换

每个子层后面都有:
  - Residual Connection (残差连接): 把输入加到子层输出上
    为什么? 解决深层网络的梯度消失问题。有了残差连接,
    梯度可以直接绕过子层传播, 让训练深层网络变得可行。
    公式: LayerNorm(x + Sublayer(x))

  - Layer Normalization: 对每个样本的 d_model 维做归一化
    为什么用 LayerNorm 而不是 BatchNorm?
    - BatchNorm 在序列维度上归一化, 但序列长度可变, 且小 batch 不稳定
    - LayerNorm 在特征维度上归一化, 与 batch 大小和序列长度无关
    - 对于 NLP 任务更天然, 因为每个 token 的特征应该独立归一化

论文中的顺序是: Sublayer → Add → LayerNorm (Post-LN)
后来很多工作(如 GPT-2, 很多实现)改为: LayerNorm → Sublayer → Add (Pre-LN)
Pre-LN 训练更稳定, 但这里我们遵循论文原始的 Post-LN 方案。
"""


class EncoderLayer(nn.Module):
    """
    Transformer 编码器的单层。
    结构: x -> Self-Attention -> Add+Norm -> FFN -> Add+Norm -> output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model:   模型维度
            num_heads: 注意力头数
            d_ff:      前馈网络中间层维度
            dropout:   Dropout 比率
        """
        super().__init__()

        # --- 子层1: Multi-Head Self-Attention ---
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # --- 子层2: Feed-Forward ---
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # --- LayerNorm ---
        # nn.LayerNorm 对最后一个维度 (d_model) 做归一化
        # eps 防止除零
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        # --- Dropout (用于子层输出的正则化) ---
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x:    (batch, seq_len, d_model) — 输入序列的表示
            mask: (batch, seq_len, seq_len)  — 可选的注意力掩码
                  编码器中通常用于 padding mask:
                  把 <pad> token 对应的位置在注意力中屏蔽掉

        返回:
            (batch, seq_len, d_model)
        """
        # --- 子层1: Self-Attention + Add & Norm ---
        # 残差连接的关键: 先把输入 x 存下来
        # attention 的输出维度与 x 相同, 所以可以直接相加
        attn_output, _ = self.self_attention(x, x, x, mask)
        # Post-LN: Sublayer → Dropout → Add → LayerNorm
        x = self.norm1(x + self.dropout1(attn_output))

        # --- 子层2: Feed-Forward + Add & Norm ---
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# ==============================================================================
# 第6步: Encoder (完整的编码器)
# ==============================================================================
"""
将 N 个 EncoderLayer 堆叠起来, 形成完整的编码器。

工作流程:
  Input Tokens → Token Embedding → + Positional Encoding
  → Dropout → [EncoderLayer × N] → Output

编码器的输出是一个上下文感知的表示序列,
每个位置的表示都融合了整个输入序列的信息。
"""


class Encoder(nn.Module):
    """
    Transformer 编码器: 堆叠的 EncoderLayer。
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=5000, dropout=0.1):
        """
        参数:
            vocab_size: 词汇表大小
            d_model:    模型维度
            num_heads:  注意力头数
            d_ff:       前馈网络中间层维度
            num_layers: EncoderLayer 的数量 (论文中是 6)
            max_len:    最大序列长度
            dropout:    Dropout 比率
        """
        super().__init__()

        # Token 嵌入层: 把离散的 token ID 映射为连续的向量
        # (vocab_size,) -> (d_model,)
        # 这是模型"认识"每个词的起点 — 语义相似的词会有相似的 embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码: 注入位置信息
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 堆叠 N 个 EncoderLayer
        # 深拷贝每个层 — 虽然结构相同, 但参数独立
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最后的 LayerNorm (很多实现会加这一层来稳定输出)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        """
        参数:
            x:    (batch, seq_len) — token ID 序列
            mask: (batch, seq_len, seq_len) — padding mask

        返回:
            (batch, seq_len, d_model) — 编码后的上下文表示
        """
        # Token Embedding
        # 注意: 论文中将 embedding 权重乘以 sqrt(d_model)
        # 这是因为 embedding 初始化的方差是 1,
        # 而位置编码的幅值在 [-1, 1] 之间,
        # 乘以 sqrt(d_model) 让两者在数值上处于相近的量级
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)

        # + Positional Encoding + Dropout
        x = self.positional_encoding(x)

        # 逐层传递
        for layer in self.layers:
            x = layer(x, mask)

        # 最终 LayerNorm
        return self.norm(x)


# ==============================================================================
# 第7步: Decoder Layer (解码器层)
# ==============================================================================
"""
一个 Decoder Layer 包含三个子层:
  1. Masked Multi-Head Self-Attention — 只能看到当前位置及之前的位置
  2. Cross-Attention — 用解码器的表示作为 query, 编码器的输出作为 key/value
  3. Position-wise FFN

Masked Self-Attention 的关键:
  在生成第 t 个 token 时, 模型不应该看到第 t+1, t+2, ... 个 token
  (因为在推理时这些 future token 还不存在!)

  实现方式: 在注意力分数矩阵的上三角设为 -inf
  (这是一个下三角矩阵: 位置 i 只能看到位置 0...i)

  scores =   [s00, s01, s02]        [s00, -inf, -inf]
             [s10, s11, s12]   →    [s10,  s11, -inf]
             [s20, s21, s22]        [s20,  s21,  s22]

Cross-Attention 的关键:
  Query 来自解码器 (我们想要生成什么),
  Key/Value 来自编码器 (输入序列提供了什么上下文信息)

  这建立了 Encoder 和 Decoder 之间的桥梁:
  解码器在每次生成时, 都会去"查阅"编码器对整句输入的理解。
"""


class DecoderLayer(nn.Module):
    """
    Transformer 解码器的单层。
    结构: x -> Masked Self-Attention -> Add+Norm
           -> Cross-Attention -> Add+Norm
           -> FFN -> Add+Norm -> output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数: 同 EncoderLayer
        """
        super().__init__()

        # --- 子层1: Masked Self-Attention ---
        # 解码器自己的 self-attention, 带因果掩码
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # --- 子层2: Cross-Attention ---
        # 解码器(query) 关注 编码器输出(key/value)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # --- 子层3: Feed-Forward ---
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # --- LayerNorm (三个子层各一个) ---
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

        # --- Dropout ---
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x:              (batch, tgt_seq_len, d_model) — 解码器输入
            encoder_output: (batch, src_seq_len, d_model) — 编码器输出
            src_mask:       padding mask for encoder (屏蔽 <pad>)
            tgt_mask:       causal mask for decoder (防看到未来)
                            + padding mask for decoder

        返回:
            (batch, tgt_seq_len, d_model)
        """
        # --- 子层1: Masked Self-Attention ---
        # 解码器自己的 self-attention
        # tgt_mask 包含:
        #   1. 因果掩码 (下三角): 防止当前位置看到未来位置
        #   2. Padding 掩码: 防止关注 <pad> token
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # --- 子层2: Cross-Attention ---
        # 解码器作为 query, 编码器输出作为 key & value
        # src_mask: 防止关注编码器中的 <pad> token
        #
        # 这里体现了 Encoder-Decoder 的协作:
        # 解码器说:"我在生成这个位置的词, 想参考一下输入序列的哪些部分"
        # 通过 cross-attention, 解码器能灵活地关注输入的不同位置
        attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout2(attn_output))

        # --- 子层3: Feed-Forward ---
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, cross_attn_weights


# ==============================================================================
# 第8步: Decoder (完整的解码器)
# ==============================================================================
"""
将 N 个 DecoderLayer 堆叠起来, 形成完整的解码器。

与编码器的不同:
  - 多了一个 Embedding 层 (因为输入输出词汇表可能不同, 但通常共享)
  - Self-Attention 使用因果掩码 (不能看未来)
  - 多了 Cross-Attention (连接编码器和解码器)
"""


class Decoder(nn.Module):
    """
    Transformer 解码器: 堆叠的 DecoderLayer。
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=5000, dropout=0.1):
        """
        参数: 同 Encoder
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x:              (batch, tgt_seq_len) — 目标 token ID 序列
            encoder_output: (batch, src_seq_len, d_model) — 编码器输出
            src_mask:       padding mask (屏蔽编码器输入中的 <pad>)
            tgt_mask:       causal mask + padding mask (解码器)

        返回:
            (batch, tgt_seq_len, d_model)
        """
        # Token Embedding (同样缩放)
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)

        # Positional Encoding
        x = self.positional_encoding(x)

        # 逐层传递
        for layer in self.layers:
            x, _ = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


# ==============================================================================
# 第9步: 完整的 Transformer 模型
# ==============================================================================
"""
现在我们把所有组件组装起来, 形成完整的 Transformer 模型。

结构回顾:

  Encoder:
    Input → Embedding + PE → [EncoderLayer × N] → Output

  Decoder:
    Target → Embedding + PE → [DecoderLayer × N] → Linear → Softmax → Prediction

  其中 DecoderLayer 内部:
    Target → Masked Self-Attn → Cross-Attn(Q=Target, K/V=Encoder Output) → FFN → Output

训练时的流程:
  1. 输入序列 进入 Encoder, 得到编码表示
  2. 目标序列(右移一位) 进入 Decoder, 同时传入编码表示
  3. Decoder 输出通过线性层 + softmax 预测下一个 token
  4. 与真实目标序列计算交叉熵损失

推理时的流程 (自回归生成):
  1. 输入序列 进入 Encoder, 得到编码表示 (只需一次)
  2. 从 <sos> 开始, 每次生成一个 token
  3. 将已生成的序列送入 Decoder, 得到下一个 token 的分布
  4. 采样或贪心选择下一个 token, 拼接到已生成序列
  5. 重复直到生成 <eos> 或达到最大长度
"""


class Transformer(nn.Module):
    """
    完整的 Transformer 模型 (Encoder-Decoder 架构)。
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, d_ff=2048, num_layers=6,
                 max_len=5000, dropout=0.1):
        """
        参数:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model:        模型维度 (论文: 512)
            num_heads:      注意力头数 (论文: 8)
            d_ff:           前馈网络隐藏层维度 (论文: 2048)
            num_layers:     编码器/解码器层数 (论文: 6)
            max_len:        最大序列长度
            dropout:        Dropout 比率 (论文: 0.1)
        """
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff,
            num_layers, max_len, dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff,
            num_layers, max_len, dropout
        )

        # 最终输出投影层: d_model -> vocab_size
        # 把解码器的表示映射到词汇表大小, 得到每个 token 的 logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # 一些实现会共享 decoder embedding 和 output projection 的权重
        # (称为 weight tying), 这里暂不实现

        # --- 参数初始化 ---
        # 论文使用 Xavier/Glorot 初始化
        # 这是深度学习中常用的初始化方法, 保持各层输出的方差稳定
        self._init_parameters()

    def _init_parameters(self):
        """使用 Xavier uniform 初始化所有参数。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def create_padding_mask(seq, pad_idx=0):
        """
        创建 padding mask: 标记序列中哪些位置是 <pad>。

        参数:
            seq: (batch, seq_len)
            pad_idx: <pad> 的 token ID

        返回:
            mask: (batch, 1, 1, seq_len)
                  True 表示有效位置, False 表示 <pad>

        用法示例:
            输入: [[1, 2, 3, 0, 0]]  (0 = <pad>)
            输出: [[[[True, True, True, False, False]]]]
        """
        # (batch, 1, 1, seq_len)
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def create_causal_mask(seq_len):
        """
        创建因果掩码 (look-ahead mask):
        确保位置 i 只能看到位置 0...i (不能看到未来)。

        返回:
            mask: (1, 1, seq_len, seq_len)
                  这是一个下三角矩阵:
                  [[1, 0, 0, 0],
                   [1, 1, 0, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1]]
        """
        # torch.triu 取上三角; diagonal=1 表示不包含对角线
        # 上三角全为 1, 我们想要 mask=0 来屏蔽, 所以取反
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        # mask: 上三角为 1 (要屏蔽的), 下三角为 0 (保留的)
        # 我们返回 1/True 表示可看到, 0/False 表示不可看到
        return (mask == 0).unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        """
        训练时的前向传播。

        参数:
            src: (batch, src_seq_len) — 源序列 token IDs
            tgt: (batch, tgt_seq_len) — 目标序列 token IDs
            src_pad_idx: 源序列的 <pad> ID
            tgt_pad_idx: 目标序列的 <pad> ID

        返回:
            logits: (batch, tgt_seq_len, tgt_vocab_size)
        """
        device = src.device

        # --- 创建掩码 ---
        # 编码器的 padding mask: 让 self-attention 忽略 <pad>
        src_mask = self.create_padding_mask(src, src_pad_idx)
        # 形状: (batch, 1, 1, src_seq_len)

        # 解码器的 padding mask (同样用于 cross-attention 的 key)
        tgt_padding_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        # 形状: (batch, 1, 1, tgt_seq_len)

        # 因果掩码: 防止解码器看到未来的 token
        tgt_seq_len = tgt.size(1)
        causal_mask = self.create_causal_mask(tgt_seq_len).to(device)
        # 形状: (1, 1, tgt_seq_len, tgt_seq_len)

        # 合并 padding mask 和 causal mask
        # 两者都满足时才为 True (才能被看到)
        # causal_mask 是 (1, 1, tgt_seq_len, tgt_seq_len)
        # tgt_padding_mask 是 (batch, 1, 1, tgt_seq_len) — key 维度上屏蔽
        tgt_mask = causal_mask & tgt_padding_mask
        # 形状: (batch, 1, tgt_seq_len, tgt_seq_len)

        # --- 前向传播 ---
        # 编码器: 输入源序列, 得到上下文表示
        encoder_output = self.encoder(src, src_mask)
        # (batch, src_seq_len, d_model)

        # 解码器: 输入目标序列 + 编码器输出, 得到解码表示
        decoder_output = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        # (batch, tgt_seq_len, d_model)

        # 输出投影: d_model -> vocab_size
        logits = self.output_projection(decoder_output)
        # (batch, tgt_seq_len, tgt_vocab_size)

        return logits

    def encode(self, src, src_pad_idx=0):
        """仅编码: 用于理解任务或缓存编码结果。"""
        src_mask = self.create_padding_mask(src, src_pad_idx)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """仅解码: 一步生成, 通常循环调用。"""
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)

    @torch.no_grad()
    def greedy_decode(self, src, max_len, start_token, end_token,
                      src_pad_idx=0):
        """
        贪心解码: 每次选择概率最高的 token 作为下一个输出。

        参数:
            src:          (1, src_seq_len) — 单个输入序列
            max_len:      最大生成长度
            start_token:  <sos> token ID
            end_token:    <eos> token ID
            src_pad_idx:  源序列的 <pad> ID

        返回:
            生成的 token 序列
        """
        self.eval()

        # 编码输入序列 (只需要一次!)
        src_mask = self.create_padding_mask(src, src_pad_idx)
        encoder_output = self.encoder(src, src_mask)

        # 从 <sos> 开始生成
        generated = torch.tensor([[start_token]], device=src.device)

        for _ in range(max_len - 1):
            # 创建解码器的掩码
            tgt_len = generated.size(1)
            tgt_mask = self.create_causal_mask(tgt_len).to(src.device)

            # 解码
            decoder_output = self.decoder(
                generated, encoder_output, src_mask, tgt_mask
            )
            # 只需要最后一个位置的输出
            logits = self.output_projection(decoder_output[:, -1, :])
            # (1, vocab_size)

            # 贪心选择概率最高的 token
            next_token = logits.argmax(dim=-1).unsqueeze(0)
            # (1, 1)

            # 拼接到已生成序列
            generated = torch.cat([generated, next_token], dim=1)

            # 如果生成了 <eos> 则停止
            if next_token.item() == end_token:
                break

        return generated.squeeze(0)  # 去掉 batch 维度


# ==============================================================================
# 第10步: 训练示例 — 复制任务 (Copy Task)
# ==============================================================================
"""
复制任务: 输入一个数字序列, 模型需要输出完全相同的序列。

这是一个经典的测试任务, 用来验证 Transformer 的基本能力:
  - 输入:   [<sos>, 3, 7, 2, 9, 5, <eos>, <pad>, ...]
  - 输出:    [3, 7, 2, 9, 5, <eos>, <pad>, ...]

虽然简单, 但它验证了:
  1. Encoder 能记住输入序列
  2. Decoder 能从 Encoder 的表示中提取信息
  3. Cross-Attention 正确地建立了 Encoder-Decoder 连接
  4. Masked Self-Attention 正确地防止了信息泄露

如果这个任务都跑不通, 说明模型实现有 bug。
"""


def generate_copy_task_data(num_samples=5000, seq_len=10, vocab_size=20):
    """
    生成复制任务的训练数据。

    词汇表:
      - 0: <pad>
      - 1: <sos> (start of sequence)
      - 2: <eos> (end of sequence)
      - 3 ~ vocab_size-1: 实际 token

    返回:
        src: 源序列 (batch, seq_len)
        tgt_input:  解码器输入 (batch, seq_len)
        tgt_output: 解码器目标 (batch, seq_len)
    """
    # 随机生成长度为 seq_len 的序列 (使用 3 到 vocab_size-1 的 token)
    # 每个序列的实际长度随机在 3 到 seq_len 之间变化
    data = torch.randint(3, vocab_size, (num_samples, seq_len))

    # 随机决定每个序列的结束位置 (在位置 2 到 seq_len 之间)
    lengths = torch.randint(2, seq_len + 1, (num_samples,))

    # 创建源序列: <sos> + data + <eos> (填充 <pad>)
    src = torch.zeros(num_samples, seq_len + 2, dtype=torch.long)
    for i in range(num_samples):
        l = min(lengths[i].item(), seq_len)
        src[i, 0] = 1  # <sos>
        src[i, 1:1+l] = data[i, :l]
        if l + 1 <= seq_len + 1:
            src[i, 1+l] = 2  # <eos>

    # 目标序列: 实际 token + <eos> (填充 <pad>)
    # tgt_input 右移一位 (前面加 <sos>)
    # tgt_output 是原始目标 (后面加 <eos>)
    tgt = torch.zeros(num_samples, seq_len + 2, dtype=torch.long)
    for i in range(num_samples):
        l = min(lengths[i].item(), seq_len)
        tgt[i, :l] = data[i, :l]
        tgt[i, l] = 2  # <eos>

    # 解码器输入: <sos> + tgt (不含最后的 <eos>, 或者保留最后)
    tgt_input = torch.zeros(num_samples, seq_len + 2, dtype=torch.long)
    for i in range(num_samples):
        l = min(lengths[i].item(), seq_len)
        tgt_input[i, 0] = 1  # <sos>
        tgt_input[i, 1:1+l+1] = tgt[i, :l+1]  # data + <eos>

    return src, tgt_input, tgt


def train_copy_task():
    """训练 Transformer 完成复制任务, 并展示中间结果。"""
    # ==============================
    # 超参数设置
    # ==============================
    VOCAB_SIZE = 20
    D_MODEL = 128        # 小模型方便快速训练
    NUM_HEADS = 4
    D_FF = 512
    NUM_LAYERS = 2
    DROPOUT = 0.1
    MAX_LEN = 50
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    # 特殊 token
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用: {device}")
    print(f"[参数] d_model={D_MODEL}, heads={NUM_HEADS}, d_ff={D_FF}, "
          f"layers={NUM_LAYERS}")

    # ==============================
    # 准备数据
    # ==============================
    print("\n[数据] 生成复制任务数据...")
    src_train, tgt_input_train, tgt_output_train = generate_copy_task_data(
        num_samples=4000, seq_len=10, vocab_size=VOCAB_SIZE
    )
    src_val, tgt_input_val, tgt_output_val = generate_copy_task_data(
        num_samples=200, seq_len=10, vocab_size=VOCAB_SIZE
    )

    # 展示一个样本
    print(f"[数据] 训练样本数: {src_train.size(0)}")
    print(f"[数据] 样本示例:")
    sample_src = src_train[0]
    sample_tgt_in = tgt_input_train[0]
    sample_tgt_out = tgt_output_train[0]
    # 过滤掉 <pad>
    src_clean = sample_src[sample_src != PAD_IDX].tolist()
    tgt_in_clean = sample_tgt_in[sample_tgt_in != PAD_IDX].tolist()
    tgt_out_clean = sample_tgt_out[sample_tgt_out != PAD_IDX].tolist()
    print(f"  源序列 (src):           {src_clean}")
    print(f"  目标输入 (tgt_input):    {tgt_in_clean}")
    print(f"  目标输出 (tgt_output):   {tgt_out_clean}")
    print(f"  (1=sos, 2=eos, 3-19=数据token)")

    # ==============================
    # 创建模型
    # ==============================
    print(f"\n[模型] 创建 Transformer...")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数量: {total_params:,}")
    print(f"[模型] 可训练参数: {trainable_params:,}")

    # ==============================
    # 损失函数与优化器
    # ==============================
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # ignore_index=PAD_IDX: 计算损失时忽略 <pad> 位置
    # 这很重要! 否则 <pad> 的损失会主导训练

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                  betas=(0.9, 0.98), eps=1e-9)
    # betas=(0.9, 0.98): 论文中的设置, 注意 β2=0.98 不是默认的 0.999

    # ==============================
    # 学习率调度器 (论文中的 Warmup 策略)
    # ==============================
    # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    # 这个调度策略: 先线性增长 (warmup), 再逐步衰减
    class TransformerLR:
        def __init__(self, optimizer, d_model, warmup_steps=4000):
            self.optimizer = optimizer
            self.d_model = d_model
            self.warmup_steps = warmup_steps
            self.current_step = 0

        def step(self):
            self.current_step += 1
            lr = self.d_model ** (-0.5) * min(
                self.current_step ** (-0.5),
                self.current_step * self.warmup_steps ** (-1.5)
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr

    scheduler = TransformerLR(optimizer, D_MODEL, warmup_steps=200)

    # ==============================
    # 训练循环
    # ==============================
    print(f"\n[训练] 开始训练 ({NUM_EPOCHS} epochs)...")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        # 随机打乱
        perm = torch.randperm(src_train.size(0))
        src_train = src_train[perm]
        tgt_input_train = tgt_input_train[perm]
        tgt_output_train = tgt_output_train[perm]

        for i in range(0, src_train.size(0), BATCH_SIZE):
            # 取一个 batch
            src_batch = src_train[i:i+BATCH_SIZE].to(device)
            tgt_in_batch = tgt_input_train[i:i+BATCH_SIZE].to(device)
            tgt_out_batch = tgt_output_train[i:i+BATCH_SIZE].to(device)

            # 前向传播
            logits = model(src_batch, tgt_in_batch, PAD_IDX, PAD_IDX)
            # logits: (batch, tgt_seq, vocab)
            # tgt_out: (batch, tgt_seq)

            # 计算损失
            # 需要 reshape 为 (batch*tgt_seq, vocab) 和 (batch*tgt_seq)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out_batch.reshape(-1)
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr = scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / (src_train.size(0) / BATCH_SIZE)

        # --- 每 5 个 epoch 做一次验证 ---
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(
                    src_val[:50].to(device),
                    tgt_input_val[:50].to(device),
                    PAD_IDX, PAD_IDX
                )
                val_loss = criterion(
                    val_logits.reshape(-1, val_logits.size(-1)),
                    tgt_output_val[:50].reshape(-1).to(device)
                )
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_loss.item():.4f} | "
                  f"LR: {lr:.6f}")

    print("=" * 70)
    print("[训练] 完成!")

    # ==============================
    # 测试模型
    # ==============================
    print("\n[测试] 用贪心解码测试复制能力...")
    print("=" * 70)

    model.eval()
    test_cases = [
        [3, 7, 12, 9],          # 使用 >=3 的 token，避免混入特殊 token
        [5, 5, 5, 5, 5],
        [4, 8, 15, 16, 17],
        [10],
        [3, 7, 9, 11, 13, 15],
    ]

    correct_results = []

    for i, test_data in enumerate(test_cases):
        # 构造输入: [<sos>, token1, token2, ..., <eos>]
        # 与训练数据格式一致
        src_list = [SOS_IDX] + test_data + [EOS_IDX]
        src = torch.tensor([src_list], dtype=torch.long).to(device)

        # 贪心解码
        output = model.greedy_decode(
            src, max_len=len(test_data)+3,
            start_token=SOS_IDX, end_token=EOS_IDX,
            src_pad_idx=PAD_IDX
        )

        # 过滤特殊 token，只保留数据 token
        output_clean = [t.item() for t in output
                        if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]

        correct = output_clean == test_data
        correct_results.append(correct)
        status = "PASS" if correct else "FAIL"
        print(f"  测试 {i+1}: 输入={test_data} -> 输出={output_clean} [{status}]")

    print("=" * 70)
    print("\n[总结]")
    if all(correct_results):
        print("所有测试 PASS! Transformer 实现正确!")
    else:
        print("部分测试 FAIL, 可能需要更多训练或调试。")
    print("模型已经学会了: 记住输入序列, 并通过 Encoder-Decoder attention "
          "逐 token 复制出来。")

    return model


# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == '__main__':
    import sys

    print("=" * 70)
    print("  Transformer 从零实现教程")
    print("  论文: Attention Is All You Need (Vaswani et al., 2017)")
    print("=" * 70)

    if '--test' in sys.argv:
        # 快速测试: 加载或训练少量 epoch
        print("\n[快速测试模式]")
        model = train_copy_task()
    else:
        # 完整训练
        model = train_copy_task()

    # ==============================
    # 注意力权重可视化 (概念演示)
    # ==============================
    print("\n[可视化] 如果想看注意力权重, 可以在 forward 中返回 attn_weights,")
    print("         用 matplotlib/seaborn 绘制热力图。")
    print("\n关键概念回顾:")
    print("  1. Self-Attention: 让序列中每个位置都能直接看到所有其他位置")
    print("  2. Multi-Head: 多个注意力子空间, 捕捉不同类型的依赖关系")
    print("  3. Positional Encoding: 给无序的 Attention 注入位置信息")
    print("  4. Residual + LayerNorm: 让深层网络稳定训练")
    print("  5. Cross-Attention: 连接编码器和解码器的桥梁")
    print("  6. Causal Mask: 保证自回归生成的因果性")
