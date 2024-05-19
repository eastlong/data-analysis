
import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
     
    distances = []
    for x_train in X_train:
        d = sqrt(np.sum((x_train - x) ** 2))
        distances.append(d)

    nearest = np.argsort(distances)
    topK_y = []
    for neighbor in nearest[:k]:
        topK_y.append(y_train[neighbor])
    
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]


