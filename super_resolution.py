import scipy.spatial
import numpy as np
query_embedding = np.array([0,0,1])
corpus_embeddings = np.array([[0,0,1],[1,0,0],[0,1,0]])
import numpy as np


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

print(cos_sim(np.array([1,0,0]),np.array([-1,0,0])))