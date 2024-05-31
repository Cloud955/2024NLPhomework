import os
import jieba
import jieba.posseg as pseg
from multiprocessing import Pool, cpu_count
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import random

# 设置文件路径
novel_dir = r'../中文语料库'  # 16部金庸小说的文件夹路径
stopwords_file = r'../test1/cn_stopwords.txt'  # 停用词文件路径

# 加载停用词
with open(stopwords_file, 'r', encoding='gbk', errors='ignore') as f:
    stopwords = set(f.read().strip().split('\n'))


# 定义只保留名词词性的函数
def filter_pos(words):
    return [word for word, flag in pseg.cut(words) if flag.startswith('n')]


# 定义预处理函数
def preprocess_text(text):
    # 分词
    words = jieba.cut(text)
    # 去除停用词
    words = [word for word in words if word not in stopwords]
    # 过滤词性
    words = filter_pos(' '.join(words))
    # 去除非中文字符
    words = [word for word in words if all('\u4e00' <= char <= '\u9fff' for char in word)]
    return words


# 处理单个文件的函数
def process_file(filepath):
    with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
        text = f.read()
    return preprocess_text(text)


# 获取所有小说文件的路径
filepaths = [os.path.join(novel_dir, filename) for filename in os.listdir(novel_dir) if filename.endswith('.txt')]

# 使用多进程处理文件
if __name__ == '__main__':
    with Pool(cpu_count()) as pool:
        processed_texts = pool.map(process_file, filepaths)

    # 输出预处理后的结果
    for i, text in enumerate(processed_texts):
        print(f"Processed text from novel {i + 1}: {' '.join(text[:100])}...")  # 仅展示前100个词

    # Word2Vec模型训练
    model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=5, sg=0, workers=cpu_count(),
                     epochs=50)
    # 保存模型
    model.save("word2vec.model")
    # 加载模型
    model = Word2Vec.load("word2vec.model")
    # 在词汇表中随机选择五组词
    words = list(model.wv.index_to_key)
    random_pairs = [(random.choice(words), random.choice(words)) for _ in range(5)]
    # 计算随机选择的词对的相似度
    similarities = []
    for word1, word2 in random_pairs:
        if word1 in model.wv and word2 in model.wv:
            similarity = model.wv.similarity(word1, word2)
            similarities.append((word1, word2, similarity))
            print(f"Similarity between '{word1}' and '{word2}': {similarity}")

    # 可视化词向量相似度
    plt.figure(figsize=(10, 6))
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=12)
    bar_width = 0.35
    bars = [f"{word1}-{word2}" for word1, word2, _ in similarities]
    heights = [similarity for _, _, similarity in similarities]
    plt.bar(bars, heights, width=bar_width)
    plt.xlabel('Word Pairs', fontproperties=font)
    plt.ylabel('Similarity', fontproperties=font)
    plt.title('Word Vector Similarity', fontproperties=font)
    plt.xticks(rotation=45, ha='right', fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.tight_layout()  # 调整布局以避免标签重叠
    plt.show()

    # 词向量聚类
    word_vectors = np.array([model.wv[word] for word in words])
    kmeans = KMeans(n_clusters=10, random_state=0).fit(word_vectors)
    labels = kmeans.labels_

    # 使用TSNE进行降维
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, n_iter=500)
    reduced_vectors = tsne.fit_transform(word_vectors)

    # 绘制聚类结果
    plt.figure(figsize=(14, 14))
    colors = ['C' + str(i) for i in range(10)]
    for i in range(len(reduced_vectors)):
        plt.scatter(reduced_vectors[i][0], reduced_vectors[i][1], c=colors[labels[i]], s=3, alpha=0.7)  # 增加点的大小和透明度
        if i % 50 == 0:  # 每隔50个点标注一次
            plt.annotate(words[i], (reduced_vectors[i][0], reduced_vectors[i][1]), fontproperties=font, fontsize=8)

    # 添加图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(10)]
    labels = [f'Cluster {i}' for i in range(10)]
    plt.legend(handles, labels)
    plt.title('Word Vector Clustering', fontproperties=font)
    plt.show()
