import os
import jieba
from collections import Counter
import math
import pandas as pd

# 定义停用词和标点符号读取函数
def load_stopwords_punctuation(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        words = {line.strip() for line in file}
    return words

# 读取停用词和标点符号
stopwords = load_stopwords_punctuation("./cn_stopwords.txt")
punctuation = load_stopwords_punctuation("./cn_punctuation.txt")
# 需要去除的多余符号
extras = stopwords.union(punctuation)

# 初始化存储信息熵的字典
entropy_data = {
    "File": [],
    "Character Entropy": [],
    "Word Entropy": [],
}

# 文件处理
input_files = "./中文语料库"
for filename in os.listdir(input_files):
    file_name_no_ext, _ = os.path.splitext(filename)  # 获取文件名
    entropy_data["File"].append(file_name_no_ext)
    # 初始化字和词的计数器
    char_counts = Counter()
    word_counts = Counter()
    # 打开文件并进行处理
    file_path = os.path.join(input_files, filename)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="ANSI", errors='replace') as file:
            txt = file.read()
            # 一元字
            char_counts.update(txt)
            # 一元词
            words = jieba.lcut(txt)
            words = [word for word in words if word not in extras]
            word_counts.update(words)

    # 计算字和词的总数
    total_char_count = sum(char_counts.values())
    total_word_count = sum(word_counts.values())

    # 计算字和词的信息熵
    char_entropy = -sum(
        (count / total_char_count) * math.log2(count / total_char_count) for count in char_counts.values())
    word_entropy = -sum(
        (count / total_word_count) * math.log2(count / total_word_count) for count in word_counts.values())

    # 添加信息熵到字典中
    entropy_data["One-gram Character"].append(char_entropy)
    entropy_data["One-gram Word"].append(word_entropy)

# 创建DataFrame对象
df = pd.DataFrame(entropy_data)

# 输出表格
print(df)
