import os
import jieba
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
import random
from os import cpu_count
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA

# 设置文件路径
novel_dir = r'../中文语料库'  # 16部金庸小说的文件夹路径
stopwords_file = r'../test1/cn_stopwords.txt'  # 停用词文件路径

# 加载停用词
def load_stopwords(path):
    with open(path, 'r', encoding='gbk', errors='ignore') as file:
        stopwords = set(file.read().split())
    return stopwords

# 预处理文本
def preprocess_text(text, stopwords):
    words = jieba.cut(text)
    filtered_words = [word for word in words if
                      word not in stopwords and all('\u4e00' <= char <= '\u9fff' for char in word)]
    return filtered_words

# 文本数据集
# 修改 Dataset 类
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回每个序列及其长度
        return [self.vocab[word] for word in self.data[idx] if word in self.vocab]

# 自定义 collate_fn 来处理不同长度的序列
def collate_fn(batch):
    batch = [torch.tensor(item, dtype=torch.long) for item in batch]
    # Pad sequences to max length in batch
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return batch



# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=50):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return hn.squeeze(0)


# 训练模型
def train_model(dataloader, model, device, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, torch.zeros_like(outputs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss {loss.item()}')

# 计算词向量之间的相似度
def calculate_similarity(embedding, word_idx1, word_idx2):
    vec1 = embedding[word_idx1].reshape(1, -1)
    vec2 = embedding[word_idx2].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# 聚类词向量
def cluster_embeddings(embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    return labels

# 可视化聚类结果
def visualize_clusters(embeddings, labels, words):
    tsne = TSNE(n_components=2, random_state=0)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 12))
    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=f'Cluster {labels[i]}')
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Clusters of Word Embeddings')
    plt.legend()
    plt.show()

# 可视化函数
def visualize_embeddings(embeddings, words, title):
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, learning_rate='auto')
    reduced = tsne.fit_transform(reduced)
    plt.figure(figsize=(12, 12))
    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    torch.cuda.empty_cache()  # 清理GPU缓存
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stopwords = load_stopwords(stopwords_file)
    filepaths = [os.path.join(novel_dir, filename) for filename in os.listdir(novel_dir) if filename.endswith('.txt')]

    # 使用线程池加速文本预处理
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        texts = list(
            executor.map(lambda fp: preprocess_text(open(fp, 'r', encoding='gbk', errors='ignore').read(), stopwords), filepaths))

    vocab = {word: i for i, word in enumerate(set(word for text in texts for word in text))}

    dataset = TextDataset(texts, vocab)
    # 使用自定义的 collate_fn
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(len(vocab), 100, 50)  # 可以尝试进一步减小模型参数
    model.to(device)
    # 获取并可视化嵌入
    embeddings = model.embedding.weight.cpu().detach().numpy()
    visualize_embeddings(embeddings, list(vocab.keys()), "PCA + t-SNE Visualization of Word Embeddings")

    # Assume other code parts are the same and embeddings have been extracted
    embeddings = model.embedding.weight.cpu().detach().numpy()
    vocab_list = list(vocab.keys())

    # Example usage of similarity calculation
    word1, word2 = '江湖', '剑客'
    idx1, idx2 = vocab[word1], vocab[word2]
    print(f"Similarity between '{word1}' and '{word2}': {calculate_similarity(embeddings, idx1, idx2)}")

    # Clustering and visualization of clusters
    labels = cluster_embeddings(embeddings)
    visualize_clusters(embeddings, labels, vocab_list)