# 2022.05.20 井上 
from ufiesia0.Config import *
print(np.__name__, 'is running in', __file__, np.random.rand(1))    

from sklearn import datasets

def get_data():
    data   = datasets.load_iris().data
    target = datasets.load_iris().target
    correct = np.eye(3)[target]
    return data, correct, target

def label_list():
    labels = datasets.load_iris().target_names
    return labels

x, c, t = get_data()
labels = label_list() 
print(x.shape, c.shape, t.shape)
print(labels)

