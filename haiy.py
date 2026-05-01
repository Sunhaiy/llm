import torch
import torch.nn as nn
from torch.nn import functional as F
#读数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#提取
print('字符总数'+str(len(text)))
chars = sorted(list(set(text)))
vocab_size=len(chars)
print('不同字符数'+str(vocab_size))

#初始化字符与数字的映射
stoi = {}
for i,chary in enumerate(chars):
    stoi[chary] = i
stio ={}
for i,chary in enumerate(chars):
    stio[i]=chary

#编解码数字
def encode(sentence):
    result = []
    for chary in sentence:
        result.append(stoi[chary])
    return result
def decode(index):
    result =[]
    for i in index:
        result.append(stio[i])
    return result
#测试
print(encode('秦牧'))
print(decode(encode('秦牧')))
#转张量
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)
#切分数据
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data.shape)
print(val_data.shape)


batch_size = 16  # 并发数
block_size = 32  # 上下文窗口
# 自动探测是否有 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        #在合法范围内完全随机取 16 个起点”，每个起点之间没有固定间隔，可能相邻、可能重复一共取 16 段，然后堆叠成二维张量。
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # 把数据搬运到模型所在的设备 (GPU) 上
    return x.to(device), y.to(device)
    
xb, yb = get_batch('train')
print("输入 x 的形状:", xb.shape) 
print("目标 y 的形状:", yb.shape)


#attention
n_embd = 128
n_head = 8   # 128 / 8 = 16，完美整除！(用 4 也可以)
n_layer = 6  # (如果你的显卡比较强，比如 4060，层数也可以加到 6)

#单头注意力
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # 简历
        q = self.query(x) # 择偶标准
        v = self.value(x) # 真实灵魂
        
        # 1. 打分机制
        # 原理：为什么除以 (k.shape[-1] ** 0.5)？
        # 因为两个矩阵相乘，维度越高，算出来的值就越大。
        # 如果分数太大，一会丢进 Softmax 里，最大的数字会变成 100%，其他的全是 0%，模型就失去了"多方参考"的能力。
        # 除以根号维度（这里是根号16=4），可以让分数变得平缓，模型就能雨露均沾。
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        
        # 2. 掩码防作弊
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 3. 变成百分比
        wei = F.softmax(wei, dim=-1)
        
        # 4. 融合输出
        out = wei @ v
        return out
class MultiHeadAttention(nn.Module):
    """多头注意力组合：4个侦探同时查案，最后汇总"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        # 把 4 个单头装进一个列表里
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 汇总后，用一个线性层把 4 个侦探的结论“搅匀”
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # 让 4 个头同时计算，然后在最后一个维度 (dim=-1) 拼装起来
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
class FeedForward(nn.Module):
    """前馈神经网络：消化关系，提纯特征"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # 先把 64 维强行拉伸到 256 维，暴露出更多微小的细节
            nn.Linear(n_embd, 4 * n_embd),
            # ReLU 会一刀切掉所有负数（过滤无用噪音）
            nn.ReLU(),
            # 把提纯后的精华重新压缩回 64 维
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    """Transformer 模块：完整的流水线层"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # 实例化多头注意力和前馈网络
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # 实例化两次归一化工具
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 先归一化，再去算注意力，最后加上最初的自己 (残差)
        x = x + self.sa(self.ln1(x))
        # 先归一化，再去深度思考，最后再次加上自己
        x = x + self.ffwd(self.ln2(x))
        return x
# =================临时测试区=================
print("\n--- 核心大脑零件测试 ---")
# 假装捏造一个批次的数据：16句话，每句32个字，每个字64维
dummy_x = torch.randn(16, 32, n_embd) 

# 拿一块 Transformer 砖头来测试
block = Block(n_embd, n_head)

# 数据穿过这块砖头
out = block(dummy_x)
print("穿过 Block 后的终极形状:", out.shape) # 期待看到 torch.Size([16, 32, 64])
class MiniGPTLanguageModel(nn.Module):
    """大模型终极外壳"""
    def __init__(self):
        super().__init__()
        # 1. 词嵌入：准备一个 3212行 x 64列 的大矩阵，存每个字的专属 64 维灵魂
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. 位置编码：准备一个 32行 x 64列 的矩阵，存 32 个座位的专属特征
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 3. 核心流水线：用 * 展开数组，串联 4 层我们刚才写的 Block
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # 4. 出厂前的最后一道归一化检查
        self.ln_f = nn.LayerNorm(n_embd) 
        
        # 5. 语言模型头 (LM Head)：把 64 维浓缩精华，反向映射成 3212 个字各自的预测得分
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 查表：把输入的数字(idx)转成词向量，并加上位置向量
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb 
        
        # 穿过 4 层大脑
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 映射成最终得分 [16, 32, 3212]
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # PyTorch 算误差要求把矩阵压扁成二维
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            # 计算预测值和标准答案的差距！
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """聊天时用来自动接龙的方法"""
        for _ in range(max_new_tokens):
            # 截断上下文，防止撑爆 32 的上限
            idx_cond = idx[:, -block_size:]
            # 预测（不传答案，不算loss）
            logits, _ = self(idx_cond)
            # 只取最后一个字的预测得分
            logits = logits[:, -1, :] 
            # 变概率
            probs = F.softmax(logits, dim=-1)
            # 抽卡！选出下一个字
            idx_next = torch.multinomial(probs, num_samples=1) 
            # 拼接到原句子里，循环继续！
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
# ==========================================
# 阶段四：正式炼丹与对话
# ==========================================

print("\n🚀 正在组装大模型引擎...")
model = MiniGPTLanguageModel()
m = model.to(device) # 推入显卡！

# 实例化 AdamW 优化器（大模型标配），学习率设为 0.001
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

max_iters = 5000 # 训练 3000 次

print("🔥 炼丹炉已点火，开始训练！(按 Ctrl+C 可跳过训练直接聊天)")
try:
    for iter in range(max_iters):
        if iter % 300 == 0:
            print(f"训练进度: {iter}/{max_iters} 步...")
            
        # 1. 发牌器抽牌
        xb, yb = get_batch('train')
        # 2. 模型试做题，算误差
        logits, loss = m(xb, yb)
        # 3. 清空上次的微积分导数
        optimizer.zero_grad(set_to_none=True)
        # 4. 极其暴力的反向传播求导！
        loss.backward()
        # 5. 更新所有参数矩阵 (W_q, W_k, W_v 等)
        optimizer.step()
        
    print(f"🎉 训练完成！最终 Loss 值降到了: {loss.item():.4f}")
except KeyboardInterrupt:
    print(f"\n⚠️ 手动中断训练！当前 Loss 值: {loss.item():.4f}")

# === 聊天前台 ===
print("\n" + "="*40)
print(" 🤖 你的手搓小模型已上线！输入 'quit' 退出。")
print("="*40)

while True:
    prompt = input("\n你: ")
    if prompt == "quit": break
    if len(prompt) == 0: continue
        
    try:
        # 中文转数字，再转张量推入显卡
        context_tensor = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        print("模型: ", end="", flush=True)
        
        # 关掉梯度计算，省显存提速
        with torch.no_grad():
            generated_ids = m.generate(context_tensor, max_new_tokens=200)[0].tolist()
        
        # 翻译回中文打印
        print(decode(generated_ids))
        
    except Exception as e:
        print(f"\n[错误]: 遇到了没见过的字 -> {e}")