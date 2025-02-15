# CIFER-10 のデータの操作の関数
# 2022.05.20 井上

from ufiesia0.Config import *
print(np.__name__, 'is running in', __file__, np.random.rand(1))    
from ufiesia0 import common_function as cf

from struct import *
import os
import time
import matplotlib.pyplot as plt

counter = [0 for i in range(10)]
path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../cifar-10-batches-bin/')) + os.sep
#path = 'C:/Python37/Lib/site-packages/cifer-10-batches-bin/'
                     
# -- ラベル --
def label_list():
    label = 'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
    return label

# CIFER-10のファイルを読出してデータを返す関数
def read_batch(filename):
    file = open(filename, 'rb') # ファイルオープン
    global data
    data = file.read()
    print('read data file:'+filename)
    file.close()
    V = np.zeros((10000, 3072))
    L = np.zeros(10000, dtype=int)
    for i in range(10000):
        offset = i * 3073    # 先頭１バイトはラベル 続く3072バイトが画像で 計3073バイト
        L[i] = data[offset]  # ラベル取得　　　　　　　　　　　　　　　　　　　　　　　　　　
        counter[int(L[i])] += 1   # 累積数カウント
        V[i][:] = np.array(unpack('3072B', data[offset+1:offset+3073])) 
    return V, L

def load_data(datapath=path):
    global x_train, t_train, x_test, t_test 
    x_train = np.zeros((50000, 3072), dtype='f4')
    t_train = np.zeros(50000, dtype=int)
    x_test  = np.zeros((10000, 3072), dtype='f4')
    t_test  = np.zeros(10000, dtype=int)
    x_train[0:10000]    , t_train[0:10000]     = read_batch(datapath + 'data_batch_1.bin')
    x_train[10000:20000], t_train[10000:20000] = read_batch(datapath + 'data_batch_2.bin')
    x_train[20000:30000], t_train[20000:30000] = read_batch(datapath + 'data_batch_3.bin')
    x_train[30000:40000], t_train[30000:40000] = read_batch(datapath + 'data_batch_4.bin')
    x_train[40000:50000], t_train[40000:50000] = read_batch(datapath + 'data_batch_5.bin')
    x_test[0:10000],      t_test[0:10000]      = read_batch(datapath + 'test_batch.bin')
    x_train = x_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1) 
    x_test  = x_test .reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    t_train = t_train.reshape(50000)
    t_test  = t_test .reshape(10000)
    return (x_train, t_train),(x_test, t_test)

def get_data(exp=False, **kwargs):
    start = time.time()
    (x_train, t_train),(x_test, t_test) = load_data()

    if exp==True:
        x_train, t_train = data_expantion(x_train, t_train)
    
    print('CIFER-10のデータが読み込まれました')
    n_train = len(x_train)
    n_test  = len(x_test)

    # -- データの軸の入替え --
    transpose = kwargs.pop('transpose', True)
    if transpose:
        print('データの軸を入れ替えます　(B, Ih, Iw, C) -> (B, C, Ih, Iw)')
        x_train = x_train.transpose(0, 3, 1, 2)
        x_test  = x_test .transpose(0, 3, 1, 2)    

    # -- データの標準化 --
    normalize = kwargs.pop('normalize', None)
    if normalize is not None:
        x_train = cf.normalize(x_train, normalize)
        x_test  = cf.normalize(x_test,  normalize)

    # -- 正解をone-hot表現に --
    y_train = np.zeros((n_train, 10), dtype='f4')
    for i in range(n_train):
        y_train[i, t_train[i]] = 1
    # -- 正解をone-hot表現に --
    y_test = np.zeros((n_test, 10), dtype='f4')
    for i in range(n_test):
        y_test [i, t_test[i]]  = 1

    print('訓練用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_train.shape, '正解値の形状', y_train.shape)    
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_train)), float(np.min(x_train))\
              , float(np.mean(x_train)), float(np.var(x_train))))
    print('評価用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_test.shape, '正解値の形状', y_test.shape)
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_test)), float(np.min(x_test))\
              , float(np.mean(x_test)), float(np.var(x_test))))

    elapsed_time = time.time() - start
    print ('elapsed_time:{0}'.format(elapsed_time) + '[sec]')
   
    return x_train, y_train, t_train, x_test, y_test, t_test

# -- サンプルの提示と順伝播の結果のテスト --
# CNNで扱うために C, Ih, Iwに入替えた場合は Ih, Iw, C に戻す必要がある
# いっぽう NNで扱う場合ベクトルにするので軸を切出す必要があるが、
# C*Ih*IwだったりIh*Iw*Cだったりすることは困る
def show_sample(data, label):
    print(data.ndim, data.shape)
    rdata = data[0] if data.ndim==4 else data
    rdata = data.reshape(3, 32, 32) if data.ndim<3 else rdata
    rdata = rdata.transpose(1, 2, 0) if rdata.shape[0]==3 else rdata 
    max_picel = np.max(rdata); min_picel = np.min(rdata) # 画素データを0～1に補正
    rdata = (rdata - min_picel)/(max_picel - min_picel)
    plt.imshow(rdata.tolist())
    plt.title(label)
    plt.show()

# -- 複数サンプルを表示(端数にも対応) --
def show_multi_samples(data, target, label_list): # data, targetは対応する複数のもの
    rdata = data.transpose(0, 2, 3, 1) if data.shape[1]==3 else data
    max_picel = np.max(rdata); min_picel = np.min(rdata) # 画素データを0～1に補正
    rdata = (rdata - min_picel)/(max_picel - min_picel)
    n_data = len(data)
    n = 50 # 一度に表示する画像数
    for j in range(0, n_data, n):   # はじめのn個、次のn個と進める
        x = rdata[j:]
        t = target[j:]
        plt.figure(figsize=(18, 10))
        m = min(n, n_data - j)      # n個以上残っていればn個、n個に満たない時はその端数
        for i in range(m):
            plt.subplot(5, 10, i+1) # 5行10列のi+1番目
            plt.imshow(x[i].tolist())
            plt.title(label_list[int(t[i])])
            plt.axis('off')
        plt.show()        

# -- 以下は実行サンプル --
'''
x_train, y_train, t_train, x_test, y_test, t_test = get_data()

print('-- データの中身を確認 train --')
while True:
    try:
        i = int(input('見たいデータの番号'))
        pick_data  = x_train[i]
        pick_label = label_list()[int(t_train[i])]    
    except:    
        print('-- テスト終了 --')
        break
    print(pick_data.shape)
    show_sample(pick_data, pick_label)
    
print('-- データの中身を確認 test --')
while True:
    try:
        i = int(input('見たいデータの番号'))
        pick_data  = x_test[i]
        pick_label = label_list()[int(t_test[i])]    
    except:    
        print('-- テスト終了 --')
        break
    show_sample(pick_data, pick_label)
#'''#
