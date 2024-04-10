# 通过中文语料库来验证Zipf's Law.
import os
import jieba
import matplotlib.pyplot as plt
from collections import Counter


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

# 初始化一个空的词频计数器
counts = Counter()

# 初始化图表
plt.figure()
# 指定要输出的排名
specified_ranks = [10, 20, 30, 40, 50]
specified_rank_results = []

# 文件处理
input_files = "./中文语料库"
for filename in os.listdir(input_files):
    file_path = os.path.join(input_files, filename)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="ANSI", errors='replace') as file:
            txt = file.read()
            words = jieba.lcut(txt)
            # 过滤掉extras中的词
            words = [word for word in words if word not in extras]
            counts.update(words)

            # 对词频进行排序
            items = counts.most_common()
            sort_list = [item[1] for item in items]  # 使用排序后的词频

            # 绘制词频分布图
            x = range(1, len(sort_list) + 1)  # 词频排名列表
            plt.plot(x, sort_list, label=filename)  # 添加图例

            # 输出指定排名的频率值和排名与频率之积
            for rank in specified_ranks:
                if rank <= len(items):
                    word, frequency = items[rank - 1]
                    product = rank * frequency
                    specified_rank_results.append((filename, rank, frequency, product))
                else:
                    specified_rank_results.append((filename, rank, None, None))

# # 设置图例位置和大小
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.legend(loc='upper right', fontsize=8)

# # 设置图表标题和坐标轴标签
# plt.title('Zipf-Law', fontsize=18)  # 标题
# plt.xlabel('Rank', fontsize=18)  # 排名
# plt.ylabel('Frequency', fontsize=18)  # 频度
#
# # 设置图表坐标轴的对数缩放
# plt.yscale('log')
# plt.xscale('log')
#
# # 保存图表并显示
# plt.savefig('./Zipf_Law.jpg')
# plt.show()
#
# # 输出指定排名的频率值和排名与频率之积
# for result in specified_rank_results:
#     filename, rank, frequency, product = result
#     if frequency is not None:
#         print(f"File: {filename}, Rank {rank}: Frequency = {frequency}, Rank * Frequency = {product}")
#     else:
#         print(f"File: {filename}, Rank {rank} is beyond the total number of words.")

#绘制图表
plt.figure()

for filename in set(result[0] for result in specified_rank_results):
    # 获取特定文件的数据
    data = [(rank, product) for file_name, rank, frequency, product in specified_rank_results if file_name == filename]
    ranks, products = zip(*data)

    # 绘制曲线并添加图例
    plt.plot(ranks, products, label=filename)

# 设置图例位置和大小
plt.legend(loc='upper right', fontsize=8)

# 设置图表标题和坐标轴标签
plt.title("词频展示", fontsize=18)  # 标题
plt.xlabel('Rank', fontsize=18)  # 排名
plt.ylabel('Rank * Frequency', fontsize=18)  # 排名 * 频度

# 保存图表并显示
plt.savefig('./Zipf_Law.jpg')
plt.show()
# 计算中文(分别以词和字为单位) 的平均信息熵