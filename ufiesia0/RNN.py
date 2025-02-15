### RNN_simple ごく基本のRNN
# 2022.11.10 A.Inoue

from ufiesia0.Config import *
from ufiesia0 import Neuron, LossFunctions
from ufiesia0 import common_function as cf
from ufiesia0.NN import NN_CNN_Base

class RNN_Base(NN_CNN_Base):
    """ RNN一般に共通の部分 """
    # 初期化以外は共通
    def __init__(self, *args, **kwargs):
        # RNN専用"
        # args : l, m, n ないし v, l, m, n
        # v:語彙数、l:語ベクトル長、m:隠れ層ニューロン数、n:出力数
        self.config = args
        loss_function_name = kwargs.pop('loss', 'MeanSquaredError')
        self.loss_function = cf.eval_in_module(loss_function_name, LossFunctions)
        self.layers = []
        self.gys, self.loss = [], []

    def reset_state(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()
        self.gys = []
        self.loss = []

def select_category(x, stocastic=False, beta=2):
    """ ダミー変数をカテゴリ変数に変換 """
    x = x.reshape(-1)
    if stocastic: # 確率的に
        p = np.empty_like(x, dtype='f4') # 極小値多数でエラーしないため精度が必要
        p[...] = x ** beta
        p = p / np.sum(p)
        y = np.random.choice(len(p), size=1, p=p)
    else:         # 確定的に
        y = np.argmax(x)
    return int(y) 

def generate(func, seed, length=200):
    """ seedに続いて一つずつ生成していく """
    gen_data = np.array(seed)
    T, l = gen_data.shape
    for j in range(length-1):
        x = gen_data[j]
        y = func(x)
        if j+1 < T: # seedの範囲は書込まない
            continue
        next_one = y.reshape(1, l)
        gen_data = np.concatenate((gen_data, next_one))
    return gen_data

def generate_text(func, x_to_id, id_to_x, length=100,
                  seed=None, stop=None, print_text=True, end='', 
                  stocastic=True, beta=2):
    """ 文字列の生成 x:文字または語 x_chain:その連なり key:次の索引キー """
    x_chain = ""
    vocab_size = len(x_to_id)
    # -- seed無指定なら辞書からランダムに選ぶ --
    if seed is None:      
        seed = random.choices(list(x_to_id)) 
        
    for j in range(length):
        # -- 書出しはseedから、その後はfunc出力から --
        if j < len(seed): # seedの範囲 
            x = seed[j]
            try:
                key = x_to_id[x]
            except:
                key = None
        else:             # seedの範囲を超えた
            key = select_category(y, stocastic, beta=beta) 
            x = id_to_x[key]
        # -- 綴る --
        x_chain += x
        if print_text:
            print(x, end=end)
        if x==stop or len(x_chain)>length:
            break
        # -- 次に備える --
        if key is not None:
            y = func(key)     # seedの範囲を含めて順伝播

    return x_chain    

def generate_text_bkup2(func, x_to_id, id_to_x, length=100,
                  seed=None, stop=None, print_text=True, end='', 
                  stocastic=True, beta=2):
    """ 文字列の生成 x:文字または語 x_chain:その連なり key:次の索引キー """
    x_chain = ""
    vocab_size = len(x_to_id)
    key = x_to_id['\n']   # keyが得られない場合に備えた初期値
    # -- seed無指定なら辞書からランダムに選ぶ --
    if seed is None:      
        seed = random.choices(list(x_to_id)) 
        
    for j in range(length):
        # -- 書出しはseedから、その後はfunc出力から --
        if j < len(seed): # seedの範囲 
            x = seed[j]
            try:
                xid = x_to_id[x]
            except:
                pass # key Errorの場合には読み飛ばす
            else:
                key = xid
        else:             # seedの範囲を超えた
            key = select_category(y, stocastic, beta=beta) 
            x = id_to_x[key]    
        # -- 綴る --
        x_chain += x
        if print_text:
            print(x, end=end)
        if x==stop or len(x_chain)>length:
            break
        # -- 次に備える --
        y = func(key)     # seedの範囲を含めて順伝播

    return x_chain    

def generate_text_bkup(func, x_to_id, id_to_x, length=100,
                  seed=None, stop=None, print_text=True, end='', 
                  stocastic=True, beta=2):
    """ 文字列の生成 x:文字または語 x_chain:その連なり key:次の索引キー """
    x_chain = ""
    vocab_size = len(x_to_id)
    # -- seed無指定なら辞書からランダムに選ぶ --
    if seed is None:      
        seed = random.choices(list(x_to_id)) 
        
    for j in range(length):
        # -- 書出しはseedから、その後はfunc出力から --
        if j < len(seed): # seedの範囲 
            x = seed[j]        
            key = x_to_id[x]   
        else:             # seedの範囲を超えた
            key = select_category(y, stocastic, beta=beta) 
            x = id_to_x[key]    
        # -- 綴る --
        x_chain += x
        if print_text:
            print(x, end=end)
        if x==stop or len(x_chain)>length:
            break
        # -- 次に備える --
        y = func(key)     # seedの範囲を含めて順伝播

    return x_chain    


class RNN_rf(RNN_Base):
    def __init__(self, l, m, n, **kwargs):
        super().__init__(l, m, n, **kwargs)
        self.rnn_layer = Neuron.RnnLayer(l, m, **kwargs)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        self.layers.append(self.rnn_layer)
        self.layers.append(self.neuron_layer)

class RNN_gf(RNN_Base):
    def __init__(self, l, m, n, **kwargs):
        super().__init__(l, m, n, **kwargs)
        self.gru_layer = Neuron.GRU(l, m, **kwargs)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        self.layers.append(self.gru_layer)
        self.layers.append(self.neuron_layer)

class RNN_lf(RNN_Base):
    def __init__(self, l, m, n, **kwargs):
        super().__init__(l, m, n, **kwargs)
        self.lstm_layer = Neuron.LSTM(l, m, **kwargs)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        self.layers.append(self.gru_layer)
        self.layers.append(self.neuron_layer)

class SimpleRNN(RNN_rf):
    def __init__(self, l, m, n, **kwargs):
        super().__init__(l, m, n, **kwargs)
    
     
class RNN_erf(RNN_Base):
    def __init__(self, v, l, m, n, **kwargs):
        super().__init__(v, l, m, n, **kwargs)
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.rnn_layer = Neuron.RnnLayer(l, m, **kwargs)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        self.layers.append(self.embedding_layer)
        self.layers.append(self.rnn_layer)
        self.layers.append(self.neuron_layer)
        
class RNN_egf(RNN_Base):
    def __init__(self, v, l, m, n, **kwargs):
        super().__init__(v, l, m, n, **kwargs)
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.gru_layer = Neuron.GRU(l, m, **kwargs)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        self.layers.append(self.embedding_layer)
        self.layers.append(self.gru_layer)
        self.layers.append(self.neuron_layer)
        
class RNN_elf(RNN_Base):
    def __init__(self, v, l, m, n, **kwargs):
        super().__init__(v, l, m, n, **kwargs)
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer = Neuron.LSTM(l, m, **kwargs)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer)
        self.layers.append(self.neuron_layer)
        
class RNN_for_Agent(RNN_Base):
    def __init__(self, l, m, n, **kwargs):
        super().__init__(l, m, n, **kwargs)
        self.rnn_layer = Neuron.RnnLayerForAgent(l, m, **kwargs)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        self.layers.append(self.rnn_layer)
        self.layers.append(self.neuron_layer)
    
    def step_only(self, x):
        """ １時刻ずつデータを処理、状態は変えない """
        y = self.rnn_layer.step_only(x)       # x.shape=B,l; y.shape=B,m 
        y = self.neuron_layer.forward(y)      # y.shape=B,n
        #print('step_only', y.shape)
        return y

    def step_and_stack(self, x, t):
        """ １時刻ずつデータを処理して状態を蓄積 """
        y = self.rnn_layer.step_and_stack(x)  # x.shape=B,l; y.shape=B,m
        y = self.neuron_layer.forward(y)      # y.shape=B,n
        losst = self.loss_function.forward(y, t)
        gy    = self.loss_function.backward() # gy.shape=B,n
        self.gys.append(gy)
        self.loss.append(losst)
        return y

    def reflect(self, eta, g_clip=None):
        """
        step_and_stackで蓄積された結果に対応して逆伝播と更新 
        rnn_layerは内部にunit毎に時系列で記憶しているものを使用
        いっぽうneuron_layerもforward時の情報が必要だが時系列に亘る記憶が無い
        そこでrnn_layerで蓄積する出力＝neuron_layer入力をアトリビュートに設定

        """
        l, m, n = self.config
        # -- 時系列とバッチを入れ替え --
        grad_y = np.array(self.gys).transpose(1, 0, 2)
        
        _, _, yr = self.rnn_layer.get_stacked_states()
        self.neuron_layer.x = yr.reshape(-1, m)

        loss = np.array(self.loss)
        # -- 逆伝播と更新 --
        self.backward(grad_y)
        self.update(eta=eta, g_clip=g_clip)
        return np.mean(loss)
