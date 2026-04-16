import torch
#读数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#提取
print('字符总数'+str(len(text)))
chars = sorted(list(set(text)))
allchar=len(chars)
print('不同字符数'+str(allchar))

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

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        #在合法范围内完全随机取 16 个起点”，每个起点之间没有固定间隔，可能相邻、可能重复一共取 16 段，然后堆叠成二维张量。
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
    
xb, yb = get_batch('train')
print("输入 x 的形状:", xb.shape) 
print("目标 y 的形状:", yb.shape)


