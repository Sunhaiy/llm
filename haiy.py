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