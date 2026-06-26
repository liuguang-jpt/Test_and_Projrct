import torch
import torch.nn as nn
import math
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from The_Time_Machine.The_Time_Machine import load_data_time_machine


# ======================== 统一模型封装 ========================
class CharRNNModel(nn.Module):
    """封装 torch.nn.RNN / GRU / LSTM 为统一接口"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, rnn_type='RNN'):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens

        self.embedding = nn.Embedding(vocab_size, embed_size)

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_size, num_hiddens, num_layers)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_size, num_hiddens, num_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.linear = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, state=None):
        # X: (batch_size, num_steps)
        X = self.embedding(X)                          # → (batch_size, num_steps, embed_size)
        X = X.permute(1, 0, 2)                         # → (num_steps, batch_size, embed_size)
        output, state = self.rnn(X, state)              # → (num_steps, batch, hiddens), state
        output = self.linear(output)                    # → (num_steps, batch, vocab_size)
        output = output.reshape(-1, output.shape[-1])   # → (num_steps*batch, vocab_size)
        return output, state

    def begin_state(self, batch_size, device):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.num_hiddens, device=device),
                    torch.zeros(self.num_layers, batch_size, self.num_hiddens, device=device))
        else:
            return torch.zeros(self.num_layers, batch_size, self.num_hiddens, device=device)


# ======================== 单轮训练 ========================
def train_epoch(model, train_iter, loss_fn, optimizer, device, use_random_iter):
    state = None
    timer = time.time()
    total_loss = total_tokens = 0

    for X, Y in train_iter:
        # ---------- 初始化 / 截断隐藏状态 ----------
        if state is None or use_random_iter:
            state = model.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(state, tuple):           # LSTM: (h, c)
                for s in state:
                    s.detach_()
            else:                                   # RNN / GRU: 单一张量
                state.detach_()

        X, Y = X.to(device), Y.to(device)
        y = Y.T.reshape(-1)                         # (num_steps * batch_size,)

        y_hat, state = model(X, state)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    perplexity = math.exp(total_loss / total_tokens)
    speed = total_tokens / (time.time() - timer)
    return perplexity, speed


# ======================== 文本生成 ========================
def predict(prefix, num_preds, model, vocab, device):
    model.eval()
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 用 prefix 预热隐藏状态
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab[y])

    # 自回归生成
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    model.train()
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# ======================== 综合对比实验 ========================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # ---------- 统一超参数 ----------
    batch_size, num_steps = 32, 35
    embed_size = 32
    num_hiddens = 256
    num_layers = 1
    lr = 1.0
    num_epochs = 200

    # ---------- 加载数据 ----------
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")

    # ---------- 三模型配置 ----------
    model_configs = {'RNN': 'RNN', 'GRU': 'GRU', 'LSTM': 'LSTM'}
    results = {}

    for name, rnn_type in model_configs.items():
        print(f"\n{'='*55}")
        print(f"  训练 {name} 中...")
        print(f"{'='*55}")

        model = CharRNNModel(vocab_size, embed_size, num_hiddens, num_layers, rnn_type)
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {total_params:,}")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        ppl_history = []
        speed_history = []
        timer_start = time.time()

        for epoch in range(num_epochs):
            ppl, speed = train_epoch(model, train_iter, loss_fn, optimizer, device,
                                     use_random_iter=False)
            ppl_history.append(ppl)
            speed_history.append(speed)

            if (epoch + 1) % 40 == 0:
                sample = predict('time traveller', 80, model, vocab, device)
                print(f"  [Epoch {epoch+1:3d}] PPL={ppl:6.1f}  |  {sample[:100]}")

        total_time = time.time() - timer_start
        final_ppl = ppl_history[-1]

        results[name] = {
            'ppl_history': ppl_history,
            'speed_history': speed_history,
            'total_time': total_time,
            'final_ppl': final_ppl,
            'total_params': total_params,
            'sample': predict('time traveller', 200, model, vocab, device),
        }

        print(f"  >>> {name} 完成: PPL={final_ppl:.1f}, 耗时={total_time:.0f}s")

    # ======================== 可视化 ========================
    colors = {'RNN': '#3498db', 'GRU': '#2ecc71', 'LSTM': '#e74c3c'}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('RNN vs GRU vs LSTM — 字符级语言模型对比', fontsize=14, fontweight='bold')

    # ---- 图 1: 困惑度曲线 ----
    ax1 = axes[0, 0]
    for name in model_configs:
        ax1.plot(results[name]['ppl_history'], color=colors[name],
                 label=f"{name} (final PPL={results[name]['final_ppl']:.1f})", linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity 变化曲线')
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 标注三段学习阶段
    ax1.axvspan(0, 40, alpha=0.05, color='yellow', label='_nolegend_')
    ax1.axvspan(40, 120, alpha=0.05, color='orange', label='_nolegend_')
    ax1.axvspan(120, 200, alpha=0.05, color='green', label='_nolegend_')
    ax1.text(20, ax1.get_ylim()[1]*0.8, '快速\n下降', ha='center', fontsize=8, color='gray')
    ax1.text(80, ax1.get_ylim()[1]*0.8, '持续\n优化', ha='center', fontsize=8, color='gray')
    ax1.text(160, ax1.get_ylim()[1]*0.8, '趋于\n收敛', ha='center', fontsize=8, color='gray')

    # ---- 图 2: 训练耗时 & 参数量 ----
    ax2 = axes[0, 1]
    names = list(model_configs.keys())
    times_vals = [results[n]['total_time'] for n in names]
    bars = ax2.bar(names, times_vals, color=[colors[n] for n in names], edgecolor='white')
    ax2.set_ylabel('训练耗时 (秒)')
    ax2.set_title('总训练时间 & 参数量')
    for bar, t, n in zip(bars, times_vals, names):
        p = results[n]['total_params']
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{t:.0f}s\n{p:,} params', ha='center', fontsize=9)

    # ---- 图 3: 最终困惑度柱状图 ----
    ax3 = axes[1, 0]
    ppl_vals = [results[n]['final_ppl'] for n in names]
    bars = ax3.bar(names, ppl_vals, color=[colors[n] for n in names], edgecolor='white')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('最终困惑度 (越低越好)')
    for bar, p in zip(bars, ppl_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{p:.1f}', ha='center', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, max(ppl_vals) * 1.3)

    # ---- 图 4: 生成文本对比 ----
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_title('最终生成文本对比', fontsize=12, fontweight='bold')
    text_lines = []
    for name in model_configs:
        sample = results[name]['sample']
        text_lines.append(f"[{name}] PPL={results[name]['final_ppl']:.1f}")
        text_lines.append(f"  {sample[:180]}...")
        text_lines.append("")
    ax4.text(0.02, 0.98, '\n'.join(text_lines), fontfamily='monospace', fontsize=7.5,
             verticalalignment='top', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    plt.tight_layout()
    plt.savefig('D:/RNN/model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: D:/RNN/model_comparison.png")
    plt.show()

    # ======================== 文本对比总结 ========================
    print(f"\n{'='*75}")
    print(f"{'模型':<8} {'最终PPL':<12} {'参数量':<12} {'耗时':<10} {'平均速度':<16}")
    print(f"{'-'*75}")
    for name in model_configs:
        r = results[name]
        avg_speed = sum(r['speed_history']) / len(r['speed_history'])
        print(f"{name:<8} {r['final_ppl']:<12.2f} {r['total_params']:<12,} "
              f"{r['total_time']:<10.1f}s {avg_speed:<16.0f} tok/s")

    print(f"\n{'='*75}")
    print("生成文本对比:")
    print(f"{'='*75}")
    for name in model_configs:
        print(f"\n--- {name} ---")
        print(results[name]['sample'])


if __name__ == '__main__':
    main()
