import os
import re
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 文件路径
novel_dir = '../中文语料库/'
punctuation_path = "../test1/cn_punctuation.txt"
stopwords_path = "../test1/cn_stopwords.txt"

# 读取并处理文本数据
data = []
for filename in os.listdir(novel_dir):
    with open(os.path.join(novel_dir, filename), "r", encoding="gbk", errors="ignore") as f:
        text = f.read()
        # 简单的文本清理
        text = re.sub(r'\s+', ' ', text)
        paragraphs = text.split("。")
        for i in range(len(paragraphs) - 1):
            prompt = paragraphs[i] + "。"
            continuation = paragraphs[i + 1]
            data.append({"prompt": prompt, "continuation": continuation})

# 数据集定义
class TextDataset(Dataset):
    def __init__(self, data, char2idx, idx2char, seq_length):
        self.data = data
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]['prompt']
        continuation = self.data[idx]['continuation']
        input_seq = [self.char2idx[char] for char in prompt]
        target_seq = [self.char2idx[char] for char in continuation]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# 简单的 Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src_embed = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_embed = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        memory = self.encoder(src_embed)
        output = self.decoder(tgt_embed, memory)
        output = self.fc(output)
        return output

# 准备数据
text = ''.join([item['prompt'] + item['continuation'] for item in data])
vocab = list(set(text))
vocab_size = len(vocab)
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

# 数据集实例化
seq_length = 10
dataset = TextDataset(data, char2idx, idx2char, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型训练
embed_size = 64
hidden_size = 128
num_heads = 4
num_layers = 2
model = TransformerModel(vocab_size, embed_size, num_heads, num_layers, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟效果较差的模型训练
num_epochs = 1  # 少量训练次数
batch_size = 32

for epoch in range(num_epochs):
    total_loss = 0
    for src_batch, tgt_batch in dataloader:
        output = model(src_batch, tgt_batch[:, :-1])
        loss = criterion(output.view(-1, vocab_size), tgt_batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')

# 生成文本
def generate_text(model, start_string, gen_length, char2idx, idx2char):
    model.eval()
    input_seq = torch.tensor([char2idx[c] for c in start_string], dtype=torch.long).unsqueeze(0)
    generated_text = start_string

    for _ in range(gen_length):
        with torch.no_grad():
            embedded_input = model.embedding(input_seq) + model.positional_encoding[:, :input_seq.size(1), :]
            memory = model.encoder(embedded_input)
            output = model.decoder(embedded_input, memory)
            output = model.fc(output)
            predicted_char_idx = output.argmax(dim=-1)[:, -1].item()
            generated_text += idx2char[predicted_char_idx]
            input_seq = torch.cat([input_seq, torch.tensor([[predicted_char_idx]], dtype=torch.long)], dim=1)

    return generated_text

# 使用模型生成文本
start_string = "郭靖沉思半晌，"
gen_length = 100
generated_text = generate_text(model, start_string, gen_length, char2idx, idx2char)
print(generated_text)
