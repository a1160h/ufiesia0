# Digits
# 2022.05.20 井上 
from ufiesia0.Config import *
print(np.__name__, 'is running in', __file__, np.random.rand(1))    

import matplotlib.pyplot as plt
from sklearn import datasets

def get_data():
    data = datasets.load_digits().data
    target = datasets.load_digits().target
    correct = np.eye(10)[target]
    data = np.array(data.tolist())
    target = np.array(target.tolist())     
    return data, correct, target

def show_sample(data, label=None):
    data = np.array(data.tolist())
    max_picel = np.max(data); min_picel = np.min(data) # 画素データを0～1に補正
    rdata = (data - min_picel)/(max_picel - min_picel)
    rdata = rdata.reshape(8, 8)
    plt.imshow(rdata.tolist(), cmap='gray')
    plt.title(label)
    plt.show()
