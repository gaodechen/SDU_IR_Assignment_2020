import json
import math
import nltk
import math
import numpy as np
import datetime


class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []     # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}    # 存储每个词及出现了该词的文档数量
        self.idf = {}   # 存储每个词的idf值
        self.k1 = 2
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)
        json_f = {"idf": self.idf,
                  "f": self.f}
        json_f = json.dumps(json_f)
        with open("model.json", 'w') as json_file:
            json_file.write(json_f)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores


def NDCG(logits, target, k):
    """
    Compute normalized discounted cumulative gain.
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :param target: 2d array [batch_size x rel_docs_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape
    assert logits.shape[1] >= k, 'NDCG@K cannot be computed, invalid value of K.'

    indices = np.argsort(-logits, 1)
    NDCG = 0
    for i in range(indices.shape[0]):
        DCG_ref = 0
        num_rel_docs = np.count_nonzero(target[i])
        for j in range(indices.shape[1]):
            if j == k:
                break
            if target[i, indices[i, j]] == 1:
                DCG_ref += 1 / np.log2(j + 2)
        DCG_gt = 0
        for j in range(num_rel_docs):
            if j == k:
                break
            DCG_gt += 1 / np.log2(j + 2)
        NDCG += DCG_ref / DCG_gt

    return NDCG / indices.shape[0]


def MRR(logits, target, k):
    """
    Compute mean reciprocal rank.
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :param target: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert logits.shape == target.shape

    # num_doc = logits.shape[1]
    indices_k = np.argsort(-logits, 1)[:, :k]  # 取topK 的index   [n, k]

    reciprocal_rank = 0
    for i in range(indices_k.shape[0]):
        for j in range(indices_k.shape[1]):
            if target[i, indices_k[i, j]] == 1:
                reciprocal_rank += 1.0 / (j + 1)
                break

    return reciprocal_rank / indices_k.shape[0]


def Precision_Recall_F1Score(logits, target, k):
    assert logits.shape == target.shape
    assert logits.shape[1] >= k, 'P_R_F1@K cannot be computed, invalid value of K.'

    num_doc = logits.shape[1]
    indices_k_ = np.argsort(-logits, 1)[:, :k]     # 取topK 的index   [n, k]
    indices_k_one_hot = np.eye(num_doc)[indices_k_]      # [n, k, num_doc]
    indices_k = np.sum(indices_k_one_hot, axis=1)      # [n, num_doc]

    precision, recall, F1_Score = [], [], []
    for i in range(logits.shape[0]):
        y_true = labels[i]
        y_pred = indices_k[i]

        # true positive
        TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
        # false positive
        # FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
        # true negative
        # TN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
        # false negative
        FN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

        P = TP / num_doc
        R = TP / (TP + FN)
        if P + R == 0:
           F1 = 0
        else:
            F1 = 2 * P * R / (P + R)
        precision.append(P)
        recall.append(R)
        F1_Score.append(F1)
    precision = sum(precision)/logits.shape[0]
    recall = sum(recall)/logits.shape[0]
    F1_Score = sum(F1_Score)/logits.shape[0]
    return precision, recall, F1_Score


def get_data():
    f_d = json.load(open("documents.json", 'r', encoding="utf-8", errors="ignore"))
    # f = json.load(open("trainingset.json", 'r'))
    # print("training set")
    # f = json.load(open("validationset.json", 'r'))
    # print("valid set")
    f = json.load(open("testset.json", 'r'))
    print("test set")

    docs, queries, labels = [], [], []
    docs_id = {}
    num = 0
    for key in f_d:
        docs.append(nltk.word_tokenize(f_d[key]))
        docs_id[key] = num
        num += 1
    num_doc = num
    print(len(docs))
    print("docs[0]:", docs[0])
    print("num_doc:", num_doc)
    for key in f["queries"].keys():
        queries.append(nltk.word_tokenize(f["queries"][key]))
        label = f["labels"][key]
        label_ = np.zeros(num_doc)
        for i in range(len(label)):
            # label[i] = docs_id[str(label[i])]
            label_[docs_id[str(label[i])]] = 1
        # one_hot = np.eye(num_doc)[label]  # [x, num_doc]
        # label = np.sum(one_hot, axis=0)  # [num_doc]
        labels.append(label_)
    labels = np.array(labels)
    # docs = ["I like dog", "table is red", "dog may eat cat and other animals"]       # [m] 每条是一个文档内容
    # queries = ["dog and cat", "table is"]    # [n] 每条是一个查询内容
    # labels = [[1, 0, 1], [0, 1, 0]]     # [n, m] 0/1 矩阵表示是否相关
    print("num_query:", len(queries))
    return docs, queries, labels    # 此处返回的是分词后结果


if __name__ == '__main__':
    tic = datetime.datetime.now()
    docs, queries, labels = get_data()
    toc = datetime.datetime.now()
    print("data preprocess finished in {}".format(toc - tic))

    tic = datetime.datetime.now()
    s = BM25(docs)
    toc = datetime.datetime.now()
    print("BM25 Model finished in {}".format(toc - tic))

    tic = datetime.datetime.now()
    scores = []
    labels = labels
    for query in queries:
        score = s.simall(query)
        scores.append(score)
        if len(scores) % 1000 == 0:
            print(len(scores))
    logits = np.array(scores)

    # indices = np.argsort(-logits, 1)[:, :10]
    # np.save("2017xxx.npy", indices)    # 最终提交文件2017xxx.npy（学号命名）
    toc = datetime.datetime.now()
    print("logits finished in {}".format(toc - tic))

    tic = datetime.datetime.now()
    ndcg_10 = NDCG(logits, labels, 10)
    print('NDCG@10 - ', ndcg_10)

    mrr = MRR(logits, labels, 10)
    print('MRR@10 - ', mrr)

    # precision, recall, F1_Score = Precision_Recall_F1Score(logits, labels, 10)
    # print("precision@10 -  ", precision, "\trecall@10 - ", recall, "\tF1_Score@10 - ", F1_Score)
    toc = datetime.datetime.now()
    print("test finished in {}".format(toc - tic))
