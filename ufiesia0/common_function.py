# common_function 縮小版
# ニューラルネットワークの構築に必要な関数
# ニューロンの核の部分以外で付帯的な機能を提供
# 2022.06.27 井上

import matplotlib.pyplot as plt
from ufiesia0.Config import *
print(np.__name__, 'is running in', __file__, np.random.rand(1))    
from ufiesia0 import LossFunctions as lf

def member_in_module(module):
    """ モジュール内のメンバ名をリスト形式で返す """
    member = []
    for d in dir(module):
        if d.startswith('__') or d=='np': # 特殊メソッドやnpは読み飛ばす
            continue
        member.append(d)
    return member    

def eval_in_module(class_name, module):
    return eval('module.'+class_name+'()')

def eval_in_module2(class_name, module, **kwargs):
    '''
    module内に定義されたクラスをclass_nameの文字列で指定してインスタンスを生成
    evalの際にはmodule内に定義されているものかをチェックするとともに、
    インスタンス化したものには元のクラスの名前をnameとして付与する
    '''
    # -- moduleに定義されたクラス名のリストを作る --　
    dir_module = [] 
    for d in dir(module):
        if d.startswith('__') or d=='np': # 特殊メソッドやnpは読み飛ばす
            continue
        dir_module.append(d)
    print('classes in module :', dir_module)
    
    # -- class_nameがモジュール内に定義されていない場合は例外発生 --
    if class_name not in dir_module: 
        raise Exception('Invalid function specified. Should be in', dir_module)

    # -- class_nameの文字列を評価してインスタンス化、元のclassから名前を継承 -- 
    function_class = eval('module.'+class_name)
    name = function_class.__name__   
    function = function_class(**kwargs)
    setattr(function, 'name', name)  
    return function

# -- 標準化 --
def normalize(data, method='standard'):
    #data = np.array(data)
    if method is None:
        pass
    elif not method:
        pass
    elif method in('0to1', 'minmax01', 'range01'):
        data_min = np.min(data); data_max = np.max(data)
        data = (data - data_min) / (data_max - data_min)  #  0～1 の範囲
        print('データは最小値=0,最大値=1に標準化されます')
    elif method in('-1to1', 'minmax-11','range-1to1'):
        data_min = np.min(data); data_max = np.max(data)
        data = (data - data_min) / (data_max - data_min)  #  0～1 の範囲
        data = data * 2 - 1                               # -1～1 の範囲
        print('データは最小値=-1,最大値=1に標準化されます')
    else: # standard
        data = (data - np.average(data)) / np.std(data)
        print('データは平均値=0,標準偏差=1に標準化されます')
    data = data.astype(Config.dtype)
    return data       

def get_accuracy(y, t, mchx=False):
    '''
    y:順伝播の結果と、対応する t:正解とを与え、分類の正解率を返す
    mchx=Trueでは正誤表も返す
    '''
    result = np.argmax(y, axis=-1)
    if y.shape == t.shape: # 正解がone_hotの場合
        correct = np.argmax(t, axis=-1)
    elif y.ndim == t.ndim + 1:
        correct = t
    else:
        raise Exception('Wrong dimension of t')
    errata = result == correct
    size = y.size / y.shape[-1] # 時系列データ対応
    accuracy = float(np.sum(errata) / size)
    if mchx:
        return accuracy, errata
    else:
        return accuracy

# -- サンプルの提示と順伝播の結果のテスト --
def test_sample(show, func, x, t, label_list=None):
    # show : 画像表示の関数, func : 順伝播の関数,  X : 入力データ, t : 正解データ
    print('\n-- テスト開始 --')
    print('データ範囲内の数字を入力、その他は終了')
    while True:
        try:
            i = int(input('テストしたいデータの番号'))
            sample_data    = x[i:i+1, :] # 次元を保存して１つ取り出す
            sample_id      = int(t.reshape(-1)[i])
            sample_label   = label_list[sample_id]  if label_list  is not None else sample_id
        except:
            print('-- テスト終了 --')
            break
        print(' -- 選択されたサンプルを表示します --')
        print('このサンプルの正解は =>', sample_label)
        if input('サンプルの画像を表示しますか？(y/n)') in ('y', 'Y'):
            show(sample_data, sample_label) # 画像の表示
        
        # サンプル表示の後にいったん問合せた方が何をやっているか分りやすい 
        if input('機械の判定を行いますか？(y/n)') in ('y', 'Y'):
            print('-- サンプルを機械で判定します --')
        else:
            print('-- テスト終了 --\n')
            break
        
        # 順伝播して結果を表示　　　　
        y = func(sample_data)             # 順伝播(入力の次元保存)

        print('ニューラルネットワークの出力\n', y)
        estimation = int(np.argmax(y))    # ニューラルネットワークの判定結果        
    
        # Noneの判定には == / != ではなく is / is not が良い(判定がうまくいく)
        if label_list is not None:
            estimate_label = label_list[estimation]
        else: 
            estimate_label = estimation            
        
        print('機械の判定は　　 　　=>', estimate_label, '\n')

# -- 誤差の記録をグラフ表示 --
def graph_for_error(*data, **kwargs):
    labels = kwargs.pop('label',  None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    legend = True
    
    if labels is None: # labelsがない場合はNoneをdata分並べる
        labels = (None,) * len(data)
        legend = False

    elif len(data)==1 and type(labels) is str:
        labels = labels,

    elif type(labels) in(list, tuple) and len(data)==len(labels):
        pass
    
    else:
        raise Exception('length of data and label mismatch.')
       
    for d, l in zip(data, labels):
        plt.plot(d, label=l)
    if legend:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()    

# -- インデクス(番号)で与えられるデータを one_hot形式に変換 --
def convert_one_hot(x, width): # x は入力、widthは one hot の幅(例えば0,1,2,3ならばwidth=4)　
    x = np.array(x)
    x_shape = x.shape
    x = x.reshape(-1)
    N = len(x)
    one_hot = np.empty((N, width), dtype=bool)
    for i, target in enumerate(x):
        one_hot[i][...]    = 0
        one_hot[i][target] = 1
    one_hot_shape = x_shape + (width,) # タプルの結合
    one_hot = one_hot.reshape(one_hot_shape)    
    #print('形状{} == 変換 ==> 形状{}'.format(x_shape, one_hot_shape))
    return one_hot

def split_train_test(*data, **kwargs):
    '''
    リストかタプルで与えたデータそれぞれをrateで指定した割合でtrainとtestに分割
    shuffle=Trueにすればランダムにtrainとtestに振分ける
    '''
    rate    = kwargs.pop('rate', None)     # 分割割合
    shuffle = kwargs.pop('shuffle', False) # 分割の際にデータをシャッフルするか否か
    seed    = kwargs.pop('seed', None)     # 乱数のシード値
    n_data  = len(data[0])                 # データ数(各データの長さ)
    
    if rate == None or rate == 0:
        return data
    elif rate<0 or rate>1:
        raise Exception('rate should be in range 0 to 1.')

    print('分割割合 =', rate, 'シャッフル =', shuffle)
   
    index = np.arange(n_data)
    if seed is not None:
        np.random.seed(seed)
    if shuffle:
        np.random.shuffle(index)

    n_test  = int(n_data * rate)
    n_train = n_data - n_test

    index_train = index[:n_train].tolist()
    index_test  = index[n_train:].tolist() 

    train, test = [], []
    for d in data:
        train.append(d[index_train])
        test.append (d[index_test])  

    return train + test # リストの結合

# -- 一般の測定用 --
class Mesurement:
    def __init__(self, model, get_acc=None):
        self.model = model
        if get_acc is not None:
            self.get_acc = get_acc
        else:
            self.get_acc = get_accuracy # get_accuracyはcommon_function内に定義
        self.error, self.accuracy = [], []
    
    def __call__(self, x, t, mchx=False):
        y = self.model.forward(x)
        l = self.model.loss_function.forward(y, t)
        l = float(l)
        self.error.append(l)
        if mchx:
            acc, errata = self.get_acc(y, t, mchx)
            self.accuracy.append(acc)
            return l, acc, errata
        else:
            acc = self.get_acc(y, t)
            self.accuracy.append(acc)
            return l, acc

    def progress(self):
        return self.error, self.accuracy

def mesurement(func, x, t):
    return Mesurement(func)(x, t)

# -- GANの測定用 --
class Mesurement_for_GAN:
    def __init__(self, model):
        self.model = model
        self.error, self.accuracy = [], []

    def get_accuracy(self, y, t):
        correct = np.sum(np.where(y<0.5, 0, 1) == t)
        return correct / len(y)

    def __call__(self, x, t): 
        y = self.model.dsc.forward(x)
        loss = self.model.loss_function.forward(y, t)
        acc  = self.get_accuracy(y, t)
        loss = float(loss)
        acc  = float(acc)
        self.error.append(loss)
        self.accuracy.append(acc)
        return loss, acc

    def progress(self, moving_average=None):
        if moving_average is None:
            return self.error, self.accuracy
        else:
            r = int(moving_average)
            n = len(self.error) - r
            err_ma = []
            acc_ma = []
            for i in range(n):
                err_ma.append(float(np.average(self.error[i:i+r])))
                acc_ma.append(float(np.average(self.accuracy[i:i+r])))
            return err_ma, acc_ma    

def graph3d(X, Y, Z, label=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    if label is not None:
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])
    ax.scatter(X, Y, Z, c='r', s=2, marker='.')
    plt.show()

def graph3d2(x, y, z, label=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if label is not None:
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])
    ax.scatter(X, Y, Z, c='r', s=2, marker='.')
    plt.show()

def data2d(rx=(0,1), ry=(0,1), n=100):
    import numpy as np
    XY =[]
    qx = np.linspace(*rx, n)
    qy = np.linspace(*ry, n)
    for x in qx:
        for y in qy:
            xy = [x, y]
            XY.append(xy)
    return np.array(XY)        

# -- 画像を生成して表示 --
def generate_random_images(func, nz, C, Ih, Iw, n=81, reverse=False, rate=1.0):
    # 画像の生成
    n_rows = int(n ** 0.5)  # 行数
    n_cols = n // n_rows    # 列数
    # 入力となるノイズ
    #noise = np.random.normal(0, 1.0, (n_rows*n_cols, nz))  # 平均 0 標準偏差 1 の乱数
    noise = np.random.randn(n_rows*n_cols, nz) * rate
    # 画像を生成して 0-1 の範囲に調整
    if C <= 1:
        #y = func(noise); print(y.shape, n, Ih, Iw)
        g_imgs = func(noise).reshape(n, Ih, Iw)
    else:
        g_imgs = func(noise).reshape(n, C, Ih, Iw).transpose(0, 2, 3, 1)
    g_imgs = (g_imgs - np.min(g_imgs)) / (np.max(g_imgs) - np.min(g_imgs))  
    g_imgs = 1 - g_imgs if reverse==True else g_imgs
    Ih_spaced = Ih + 2; Iw_spaced = Iw + 2
    if C <= 1:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols))  # 全体の画像
    else:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols, C)) # 全体の画像
    matrix_image[...] = 1.0 if reverse==True else 0.0 
    #print(matrix_image.shape, g_imgs.shape)
    #  生成された画像を並べて一枚の画像にする
    for i in range(n_rows):
        for j in range(n_cols):
            g_img = g_imgs[i*n_cols + j]
            top  = i*Ih_spaced
            left = j*Iw_spaced
            matrix_image[top : top+Ih, left : left+Iw] = g_img

    plt.figure(figsize=(9, 9))
    if C <=1:
        plt.imshow(matrix_image.tolist(), cmap='Greys_r')
    else:
        plt.imshow(matrix_image.tolist())#, cmap='Greys_r')
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  # 軸目盛りのラベルと線を消す
    plt.show()

def split_by_janome(text):
    """ janomeの形態素分析で文章を語に分割 """
    from janome.tokenizer import Tokenizer
    word_list = Tokenizer().tokenize(text, wakati=True)
    return list(word_list)
    
def preprocess(x_list, corpus=[], x_to_id={}, id_to_x={}):
    """ コーパスと変換辞書作成 """
    for x in x_list:
        if x not in x_to_id:
            new_id = len(x_to_id)
            x_to_id[x] = new_id
            id_to_x[new_id] = x
        corpus.append(x_to_id[x])
    return corpus, x_to_id, id_to_x    

# -- 日本語文章を文字単位で corpus と辞書に変換 ---------
#    既存のcorpusと辞書を渡すと辞書に追加して corpus を拡張
def preprocess_jp(text, corpus=[], char_to_id={}, id_to_char={}):
    corpus = corpus.tolist() if type(corpus)==np.ndarray else corpus
    #corpus=np.array(corpus).tolist()     # リスト,arrayの両方に対応
    char_list = sorted(list(set(text)))  # 文字リスト作成、setで文字の重複をなくす
    #print('テキストの文字数:', len(text))          # len() で文字列の文字数を取得
    #print('文字数（重複無し）:', len(char_list))

    for char in text:
    #for char in char_list:     
        if char not in char_to_id:
            new_id = len(char_to_id)
            char_to_id[char] = new_id
            id_to_char[new_id] = char

    # corpusへの変換
    for char in text:
        corpus.append(char_to_id[char])
    #corpus = np.array([char_to_id[char] for char in text]) # 追加はできないからNG   
    return corpus, char_to_id, id_to_char    

def print_jp_texts(data, id_to_char={}):
    for d in data:
        print(id_to_char[int(d)], end='')
    print('\n')    

# -- 文章などの一続きの学習データを入力データと正解データとして切り出す ------------ 
def arrange_time_data(corpus, time_size, CPT=None, step=None):
    print('一つながりのデータから時系列長の入力データとそれに対する正解データを切り出します')
    data = []
    # キャプチャ幅も切出し間隔も指定されない場合はいずれもtime_size    
    if CPT is None and step is None:
        CPT = step = time_size
    # キャプチャ幅が指定され、切出し間隔が指定されない場合はCPTに合わせる
    if CPT is not None and step is None: 
        step = CPT
    # キャプチャ幅は指定されないが、切り出し幅が指定された場合
    if CPT is None and step is not None:
        CPT = time_size

    print('時系列長は',time_size, 'データの切出し間隔は', step, 'です')
    print('正解データの時系列長は', CPT, 'です')        
    for i in range(0, len(corpus) - time_size, step):   # 時系列長＋１の長さのデータを一括して
        data.append(corpus[i : i + time_size + 1])      # step幅ずつずらして切出す
    data = np.array(data, dtype='int')
    input_data   = data[:, 0:time_size]                 # 0～time_size-1 が入力　
    correct_data = data[:, time_size-CPT+1:time_size+1] # 1～time_size(1時刻ずらし)が正解
    if CPT == 1:
        correct_data = correct_data.reshape(-1)
    print('入力データと正解値の形状：', input_data.shape, correct_data.shape)
    return input_data, correct_data
