from ufiesia0.Config import *
from ufiesia0 import common_function as cf
print(np.__name__, 'is running in', __file__, np.random.rand(1))    
import matplotlib.pyplot as plt

path = __file__ + '\../../cifar-10-batches-py/'
print(path) 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_batch(file):
    print(file)
    dict = unpickle(file)
    #print(dict.keys())
    data   = dict[b'data']
    labels = dict[b'labels']
    return data, labels

def load_data(datapath=''):
    data1, labels1 = read_batch(datapath + 'data_batch_1')
    data2, labels2 = read_batch(datapath + 'data_batch_2')
    data3, labels3 = read_batch(datapath + 'data_batch_3')
    data4, labels4 = read_batch(datapath + 'data_batch_4')
    data5, labels5 = read_batch(datapath + 'data_batch_5')
    x_train = np.vstack((data1, data2, data3, data4, data5))
    t_train = np.hstack((labels1, labels2, labels3, labels4, labels5)).astype('uint8')
    x_test, t_test = read_batch(datapath + 'test_batch')
    x_test = np.array(x_test)
    t_test = np.array(t_test, dtype='uint8')

    return x_train, t_train, x_test, t_test

def label_list():
    label = ['airplane',   # 0
             'automobile', # 1
             'bird',       # 2
             'cat',        # 3
             'deer',       # 4
             'dog',        # 5
             'frog',       # 6
             'horse',      # 7
             'ship',       # 8
             'truck']      # 9
    return label

def get_data(path = path, **kwargs):
    x_train, t_train, x_test, t_test = load_data(path)
    print('CIFER-10のデータが読み込まれました')

    # -- データの軸の切り出し --
    if kwargs.pop('image', False):
        x_train = x_train.reshape(-1, 3, 32, 32)
        x_test  = x_test .reshape(-1, 3, 32, 32)
    
    # -- データの軸の入替え --
    if kwargs.pop('transpose', False):
        print('データの軸を画像表示用に入れ替えます (B, C, Ih, Iw) -> (B, Ih, Iw, C)')
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test  = x_test .transpose(0, 2, 3, 1)    

    # -- データの標準化 --
    normalize = kwargs.pop('normalize', True)
    if normalize is not None: # True, 0to1, -1to1 など指定可能 
        x_train = cf.normalize(x_train, normalize)
        x_test  = cf.normalize(x_test,  normalize)

    # -- 正解をone-hot表現に --
    c_train = np.eye(10)[t_train].astype(Config.dtype)
    c_test  = np.eye(10)[t_test] .astype(Config.dtype)    

    print('訓練用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_train.shape, '正解値の形状', c_train.shape)    
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_train)), float(np.min(x_train))\
              , float(np.mean(x_train)), float(np.var(x_train))))
    print('評価用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_test.shape, '正解値の形状', c_test.shape)
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_test)), float(np.min(x_test))\
              , float(np.mean(x_test)), float(np.var(x_test))))

    return x_train, c_train, t_train, x_test, c_test, t_test

# -- サンプルの提示と順伝播の結果のテスト --
# CNNで扱う場合は C, Ih, Iw が良く　
# 画像表示には　　Ih, Iw, C が良い
# NNで扱う場合はベクトルだが C*Ih*Iw Ih*Iw*C どちらか不明
def show_sample(data, label):
    #print('### debug', data.shape)
    rdata = data[0] if data.ndim==4 else data
    rdata = data.reshape(3, 32, 32) if data.ndim<=2 else rdata
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
if __name__=='__main__':

    x_train, c_train, t_train, x_test, c_test, t_test = get_data()#normalize=True)

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

