# 2024NLPhomework
数据集来源：https://share.weiyun.com/5zGPyJX，将压缩包解压重命名为“中文语料库”置于项目下

停词和标点符号文件来源：https://github.com/zcqin/DLNLP2023/blob/main/cn_stopwords.txt，下载置于项目下

zipf_law:根据数据集验证zipf_law定律

entropy:计算数据集中各文件的字/词信息熵

word2vec：基于pytorch，利用给定语料库，利用Word2Vec来训练词向量，通过计算词向量之间的语意距离、某一类词语的聚类、某些段落直接的语意关联、类比推理任务等方法来验证词向量的有效性。

lstm：利用LSTM来训练词向量，通过计算词向量之间的语意距离和词语的聚类的方法来验证词向量的有效性。
