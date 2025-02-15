def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_batch(file):
    print(file)
    dict = unpickle(file)
    print(dict.keys())
    data = dict[b'data']
    labels = dict[b'labels']
    return data, labels

def load_data(datapath=''):
    import numpy as np
    data1, labels1 = read_batch(datapath + 'data_batch_1')
    data2, labels2 = read_batch(datapath + 'data_batch_2')
    data3, labels3 = read_batch(datapath + 'data_batch_3')
    data4, labels4 = read_batch(datapath + 'data_batch_4')
    data5, labels5 = read_batch(datapath + 'data_batch_5')
    x_train = np.vstack((data1, data2, data3, data4, data5))
    t_train = np.hstack((labels1, labels2, labels3, labels4, labels5))
    x_test, t_test = read_batch(datapath + 'test_batch')
    t_test = np.array(t_test)

    x_train = x_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1) 
    x_test  = x_test .reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

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

x_train, t_train, x_test, t_test = load_data()
print(x_train.shape, type(x_train))
print(t_train.shape, type(t_train))
print(x_test.shape, type(x_test))
print(t_test.shape, type(t_test))

