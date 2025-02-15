# RNN
# Base functions for RNN and each RNN is defined
# 2022.09.06 A.Inoue
from ufiesia0.Config import *
from ufiesia0 import Neuron
from ufiesia0 import LossFunctions
from ufiesia0 import common_function as cf
import copy

# ニューラルネットワークの構築
class RNN_Base: 
    def __init__(self, *args, **kwargs):
        # 引数の判定(configで渡すだけだが一応見ておく,後日lやnをデータから判定)
        if len(args) == 3:
            l, m, n = args
            self.config = l, m, n    # 入力数,隠れ層ニューロン数,出力数
        elif len(args) == 4:
            v, l, m, n = args
            self.config = v, l, m, n # 語彙数,語ベクトル長,隠れ層ニューロン数,出力数

        # 出力層(full_connection_layer、affine_layer)
        ol_act_cand = 'Sigmoid' if n == 1 else 'Softmax'
        opt_for_ol = {}
        opt_for_ol['activate'] = kwargs.pop('ol_act', ol_act_cand)
        opt_for_ol.update(kwargs)    # kwargsに残ったものを結合
        self.opt_for_ol = opt_for_ol # subclassで出力層初期化時に使用

        self.layers = []
        self.forward_options = None  # RNN全体としてのCPTやDOの設定

        # 損失関数
        loss_function_name = kwargs.pop('loss_function', 'MeanSquaredError')
        ignore_label = kwargs.pop('ignore', -1)
        print(loss_function_name, ignore_label)
        self.loss_function = cf.eval_in_module(loss_function_name, LossFunctions)

        # 重み共有
        share_weight = kwargs.pop('share_weight', False) # Embeddingと全結合層の重み共有 
        if share_weight == True and l==m and v==n:       # 重み共有の場合には、D==H の必要がある
            print('embeddingとoutputで重みが共有されます')
            self.share_weight = True
        else:                                            
            self.share_weight = False

    def summary(self):
        print('～～ model summary of ' + str(self.__class__.__name__) + str(self.config)
              + ' ～～～～～～～～～～～～')
        for i, layer in enumerate(self.layers):
            print('layer', i, layer.__class__.__name__)
            print(' configuration =', layer.config, end='')
            if hasattr(layer, 'method'):
                print('\n method =', layer.method, end=' ')
            if hasattr(layer, 'activator'):
                print('\n activate =', layer.activator.__class__.__name__, end=' ')
            if hasattr(layer, 'update'):
                print(' optimize =', layer.optimizer_w.__class__.__name__, end='')
            print('\n------------------------------------------------------------------------')
        if hasattr(self, 'loss_function'):
            print(' loss_function =', self.loss_function.__class__.__name__)
        print('～～ end of summary ～～～～～～～～～～～～～～～～～～～～～～～～～～～～\n')

    def reset_state(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()

    def set_state(self, r0): # 最後のRNN層のr0をセット Seq2seqのdecoderで使う
        for layer in reversed(self.layers):
        #for layer in self.layers:  # 最初のRNN層ではうまく行かない
            if hasattr(layer, 'set_state'):
                layer.set_state(r0)
                break

    def get_grad_r0(self):
        for layer in reversed(self.layers):
            if hasattr(layer, 'grad_r0'):
                return layer.grad_r0

    def update(self, **kwargs):
        # 始めと終わりの層は重み共有を判定して更新
        if self.share_weight and hasattr(self, 'embedding_layer'):
            # embedding_layerをoutput_layerの勾配を加味して更新
            self.layers[0].grad_w += self.layers[-1].grad_w.T
            self.layers[0].update(**kwargs)
            # output_layerはembedding_layerのwに合わせる
            self.layers[-1].w = self.layers[0].w.T
        else:
            self.layers[0].update(**kwargs)
            self.layers[-1].update(**kwargs)
        # 重み共有に関係ない層を一括更新    
        for layer in self.layers[1:-1]:
            if hasattr(layer, 'update'):
                layer.update(**kwargs)

    def forward(self, x, t=None):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
            #print('### debug', layer.name, opt)

        if t is None:
            return y
        elif hasattr(self, 'loss_function'):
            #print('### debug1', t.shape, y.shape)
            l = self.loss_function.forward(y, t)
            return y, l
        else:
            raise Exception("Can't get loss by forward.")

    # -- 逆伝播 --
    def backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            raise Exception("Can't get gradient for backward." \
                            , 'gy =', gy, 'gl =', gl)
        #gy = gy.reshape(len(gy), -1)              # バッチサイズ分拡大
        gx = self.layers[-1].backward(gy) 
        for layer in reversed(self.layers[:-1]):
            gx = layer.backward(gx)
        return gx

    def loss(self, y, t):
        if hasattr(self, 'loss_function'):
            l = self.loss_function.forward(y, t)
            return l
        else:
            raise Exception('No loss_function defined.')
        
    def generate(self, seed, leng=200,
            beta=2, skip_ids=None, end_id=None, stocastic=True, *args):
        """ seed から順に次時刻のデータを leng になるまで生成 """
        if hasattr(self, 'embedding_layer'):
            return self.generate_id_from_id(seed, leng, beta, skip_ids, end_id, stocastic)
        else:
            return self.generate_data_from_data(seed, leng)

    def generate_data_from_data(self, seed, length=200):
        """
        seed から順に次時刻のデータを size になるまで生成(画像などの値) 
        embedding を伴わない RNN で画像など、値から値を生成する場合に適合

        """
        T, l = seed.shape
        gen_data = np.zeros((length, l))
        gen_data[:T] = seed
        self.reset_state()
        for j in range(length - 1):
            x = gen_data[j]
            y = self.forward(x.reshape(1, 1, l))
            if j+1 < T: # seedの範囲は書込まない
                continue
            gen_data[j+1] = y.reshape(-1)
        return gen_data

    def generate_id_from_id(self, seed, length=200,
               beta=2, skip_ids=None, end_id=None, stocastic=True):
        """
        seed から順に次時刻のデータを length になるまで生成(文字列などのid)
        embedding を備えた RNN で文字列生成をする場合に適合

        """
        T = len(seed)
        gen_data = np.array(seed)
        self.reset_state()
        for j in range(length - 1):
            x = gen_data[j]
            y = self.forward(x.reshape(1, 1)) # embeddingへの入力形状は(B, T) 
            if j+1 < T:   # seedの範囲はgen_dataに出力を加えない
                continue
            y = y.reshape(-1)
            if stocastic: # 確率的に
                p = np.empty_like(y, dtype='f4') # 極小値多数でエラーしないため精度が必要
                p[...] = y ** beta
                p = p / np.sum(p)
                next_one = np.random.choice(len(p), size=1, p=p)
            else:         # 確定的に
                next_one = np.argmax(y)
            if end_id is not None and next_one==end_id: # end_idが出現したら打切り　
                break
            if (skip_ids is None) or (next_one not in skip_ids):
                gen_data = np.concatenate((gen_data, next_one))
        return gen_data

    # -- seed から順に次時刻のデータを size になるまで生成(画像などの値) --
    # embedding を伴わない RNN で画像など、値から値を生成する場合に適合
    def generate_data_from_data2(self, seed, length=200,
                                verbose=False, extension=False):
        """ 末尾に一つ追加しながら頭は変えずに伸ばしていく """
        T, l = seed.shape
        gen_data = np.zeros((length, l))
        gen_data[0:T, :] = seed
        x_record, y_record = [], []
        for j in range(length - T):                   
            self.reset_state()
            x = gen_data[0:j+T, :] if extension else gen_data[j:j+T, :]
            y = self.forward(x.reshape(1, -1, l))
            gen_data[j+T, :] = y[0, -1, :]
            x_record.append(x)
            y_record.append(y)
        if verbose:
            return gen_data, x_record, y_record
        return gen_data

    # -- seed から順に次時刻のデータを length になるまで生成(文字列などのid) --
    # embedding を備えた RNN で文字列生成をする場合に適合
    def generate_id_from_id2(self, seed, length=200,
               beta=2, skip_ids=None, end_id=None, stocastic=True, extension=False):
        """ 末尾に一つ追加しながら頭は変えずに伸ばしていく """
        T = len(seed)
        gen_data = np.array(seed)
        for i in range(length - T):
            self.reset_state()
            x = gen_data if extension else gen_data[i:]
            y = self.forward(x.reshape(1, -1))
            y = y[0,-1,:]                 # 非バッチ処理の末尾の時刻
            if stocastic: # 確率的に
                p = np.empty_like(y, dtype='f4') # 極小値多数でエラーしないため精度が必要
                p[...] = y ** beta
                p = p / np.sum(p)
                next_one = np.random.choice(len(p), size=1, p=p)
            else:         # 確定的に
                next_one = np.argmax(y)
            if end_id is not None and next_one==end_id:  # end_idが出現したら打切り　
                break
            if (skip_ids is None) or (next_one not in skip_ids):
                gen_data = np.concatenate((gen_data, next_one))

        return gen_data

    # -- パラメタから辞書 --
    def export_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                params['layer'+str(i)+'_w'] = np.array(layer.w)
            if hasattr(layer, 'v'):
                params['layer'+str(i)+'_v'] = np.array(layer.v)
            if hasattr(layer, 'b'):
                params['layer'+str(i)+'_b'] = np.array(layer.b)
        return params

    # -- 辞書からパラメタ --
    def import_params(self, params):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                layer.w = np.array(params['layer'+str(i)+'_w']) 
            if hasattr(layer, 'v'):
                layer.v = np.array(params['layer'+str(i)+'_v']) 
            if hasattr(layer, 'b'):
                layer.b = np.array(params['layer'+str(i)+'_b'])
            
    # -- 学習結果の保存 --
    def save_parameters(self, file_name):
        title = self.__class__.__name__
        cf.save_parameters(file_name, title, self.export_params())

    # -- 学習結果の継承 --
    def load_parameters(self, file_name):
        title_f, params = cf.load_parameters(file_name) 
        title = self.__class__.__name__
        if title == title_f:
            self.import_params(params)
            print('パラメータが継承されました')
        else:
            print('!!構成が一致しないためパラメータは継承されません!!')
        return params


# -- 外部から直接アクセス --
def build(Class, *args, **kwargs):
    print(Class, args, kwargs)
    global model
    model = Class(*args, **kwargs)
    
def summary():
    model.summary()

def reset_state():
    model.reset_state()
    
def forward(*args, **kwargs):#d x, t=None, DO_rate=0.0, CPT=None):
    return model.forward(*args, **kwargs)#x, t, DO_rate, CPT)

def backward(*args, **kwargs):#grad_y=None, gl=1):
    model.backward(*args, **kwargs)#grad_y, gl)

def update(**kwargs):
    model.update(**kwargs)

def loss(y, t):
    return model.loss(y, t)

def loss_function(y, t):
    return model.loss(y, t)

def generate(seed, length=200,
        beta=2, skip_ids=None, end_id=None, stocastic=True, verbose=False, extension=False):
    return model.generate(seed, length,
                          beta, skip_ids, end_id, stocastic, verbose, extension)

# -- 学習結果の保存 --
def save_parameters(file_name):
    model.save_parameters(file_name)
    
# -- 学習結果の継承 --
def load_parameters(file_name):
    return model.load_parameters(file_name)

class RNN_rf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_rf'
        super().__init__(*args, **kwargs)
        l, m, n = self.config
        # -- 各層の初期化 --
        self.rnn_layer    = Neuron.RNN(l, m, **kwargs)
        self.output_layer = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.rnn_layer)
        self.layers.append(self.output_layer)

class RNN_lf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_lf'
        super().__init__(*args, **kwargs)
        l, m, n = self.config
        # -- 各層の初期化 --
        self.lstm_layer   = Neuron.LSTM(l, m, **kwargs)
        self.output_layer = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.lstm_layer)
        self.layers.append(self.output_layer)

class RNN_gf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_gf'
        super().__init__(*args, **kwargs)
        l, m, n = self.config
        # -- 各層の初期化 --
        self.gru_layer    = Neuron.GRU(l, m, **kwargs)
        self.output_layer = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.gru_layer)
        self.layers.append(self.output_layer)

class RNN_rrf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_rrf'
        super().__init__(*args, **kwargs)
        l, m, n = self.config
        # -- 各層の初期化 --
        self.rnn_layer1   = Neuron.RNN(l, m, **kwargs)
        self.rnn_layer2   = Neuron.RNN(m, m, **kwargs)
        self.output_layer = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.rnn_layer1)
        self.layers.append(self.rnn_layer2)
        self.layers.append(self.output_layer)

class RNN_llf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_llf'
        super().__init__(*args, **kwargs)
        l, m, n = self.config
        # -- 各層の初期化 --
        self.lstm_layer1  = Neuron.LSTM(l, m, **kwargs)
        self.lstm_layer2  = Neuron.LSTM(m, m, **kwargs) 
        self.output_layer = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.lstm_layer1)
        self.layers.append(self.lstm_layer2)
        self.layers.append(self.output_layer)

class RNN_ggf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_ggf'
        super().__init__(*args, **kwargs)
        l, m, n = self.config
        # -- 各層の初期化 --
        self.gru_layer1   = Neuron.GRU(l, m, **kwargs)
        self.gru_layer2   = Neuron.GRU(m, m, **kwargs) 
        self.output_layer = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.gru_layer1)
        self.layers.append(self.gru_layer2)
        self.layers.append(self.output_layer)

class RNN_erf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_erf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.rnn_layer       = Neuron.RNN(l, m, **kwargs)
        self.output_layer    = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.rnn_layer)
        self.layers.append(self.output_layer)
        
class RNN_elf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_elf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer      = Neuron.LSTM(l, m, **kwargs)
        self.output_layer    = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer)
        self.layers.append(self.output_layer)
        
class RNN_egf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_egf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.gru_layer       = Neuron.GRU(l, m, **kwargs)
        self.output_layer    = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.gru_layer)
        self.layers.append(self.output_layer)
        
class RNN_ellf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_ellf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer1     = Neuron.LSTM(l, m, **kwargs)
        self.lstm_layer2     = Neuron.LSTM(m, m, **kwargs) 
        self.output_layer    = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer1)
        self.layers.append(self.lstm_layer2)
        self.layers.append(self.output_layer)
        
class RNN_eggf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_eggf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.gru_layer1     = Neuron.GRU(l, m, **kwargs)
        self.gru_layer2     = Neuron.GRU(m, m, **kwargs) 
        self.output_layer    = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.gru_layer1)
        self.layers.append(self.gru_layer2)
        self.layers.append(self.output_layer)
        
class RNN_elllf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_elllf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer1     = Neuron.LSTM(l, m, **kwargs)
        self.lstm_layer2     = Neuron.LSTM(m, m, **kwargs) 
        self.lstm_layer3     = Neuron.LSTM(m, m, **kwargs) 
        self.output_layer    = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer1)
        self.layers.append(self.lstm_layer2)
        self.layers.append(self.lstm_layer3)
        self.layers.append(self.output_layer)
        
class RNN_ellllf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_ellllf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer1     = Neuron.LSTM(l, m, **kwargs)
        self.lstm_layer2     = Neuron.LSTM(m, m, **kwargs) 
        self.lstm_layer3     = Neuron.LSTM(m, m, **kwargs) 
        self.lstm_layer4     = Neuron.LSTM(m, m, **kwargs) 
        self.output_layer    = Neuron.NeuronLayer(m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer1)
        self.layers.append(self.lstm_layer2)
        self.layers.append(self.lstm_layer3)
        self.layers.append(self.lstm_layer4)
        self.layers.append(self.output_layer)

class RNN_el(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_el'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer      = Neuron.LSTM(l, m, **kwargs)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer)

class RNN_ell(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_ell'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer1     = Neuron.LSTM(l, m, **kwargs)
        self.lstm_layer2     = Neuron.LSTM(m, m, **kwargs)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer1)
        self.layers.append(self.lstm_layer2)

class RNN_elaf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_elaf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer      = Neuron.LSTM(l, m, **kwargs)
        self.attention_layer = Neuron.Attention()
        self.output_layer    = Neuron.NeuronLayer(2*m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer)
        self.layers.append(self.attention_layer)
        self.layers.append(self.output_layer)

    def forward(self, x, z, DO_rate=0.0, CPT=None):
        y = self.embedding_layer.forward(x, DO_rate)
        y = self.lstm_layer.forward(y, DO_rate, CPT)
        c = self.attention_layer.forward(z, y)     # z:key&value, y:query 
        y = np.concatenate((c, y), axis=2)
        y = self.output_layer.forward(y)
        return y

    def backward(self, gy):
        gx = self.output_layer.backward(gy)
        B, T, H2 = gx.shape; H = H2//2
        gz, gq = self.attention_layer.backward(gx[:,:,:H]) # gz:key&valueの勾配、gq:queryの勾配
        gr = gx[:,:,H:] + gq                       # forwardでattentionとaffineに配っていることに対応
        gx = self.lstm_layer.backward(gr)
        gx = self.embedding_layer.backward(gx)
        gr0 = self.lstm_layer.grad_r0
        return gz, gr0
        
class RNN_ellaf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_elaf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer1     = Neuron.LSTM(l, m, **kwargs)
        self.lstm_layer2     = Neuron.LSTM(m, m, **kwargs)
        self.attention_layer = Neuron.Attention()
        self.output_layer    = Neuron.NeuronLayer(2*m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer1)
        self.layers.append(self.lstm_layer2)
        self.layers.append(self.attention_layer)
        self.layers.append(self.output_layer)

    def forward(self, x, z, DO_rate=0.0, CPT=None):
        y = self.embedding_layer.forward(x, DO_rate)
        y = self.lstm_layer1.forward(y, DO_rate)
        y = self.lstm_layer2.forward(y, DO_rate, CPT)
        c = self.attention_layer.forward(z, y)     # z:key&value, y:query 
        y = np.concatenate((c, y), axis=2)
        y = self.output_layer.forward(y)
        return y

    def backward(self, gy):
        gx = self.output_layer.backward(gy)
        B, T, H2 = gx.shape; H = H2//2
        gz, gq = self.attention_layer.backward(gx[:,:,:H]) # gz:key&valueの勾配、gq:queryの勾配
        gr = gx[:,:,H:] + gq                       # forwardでattentionとaffineに配っていることに対応
        gx = self.lstm_layer2.backward(gr)
        gx = self.lstm_layer1.backward(gx)
        gx = self.embedding_layer.backward(gx)
        gr0 = self.lstm_layer2.grad_r0
        return gz, gr0
        
class RNN_elsaf(RNN_Base):
    def __init__(self, *args, **kwargs):
        self.title = 'RNN_elaf'
        super().__init__(*args, **kwargs)
        v, l, m, n = self.config
        # -- 各層の初期化 --
        self.embedding_layer = Neuron.Embedding(v, l, **kwargs)
        self.lstm_layer      = Neuron.LSTM(l, m, **kwargs)
        self.attention_layer = Neuron.Attention()
        self.output_layer    = Neuron.NeuronLayer(2*m, n, **self.opt_for_ol)
        # -- layerのまとめ --
        self.layers.append(self.embedding_layer)
        self.layers.append(self.lstm_layer)
        self.layers.append(self.attention_layer)
        self.layers.append(self.output_layer)

    def forward(self, x, DO_rate=0.0, CPT=None):
        z = self.embedding_layer.forward(x, DO_rate)
        z = self.lstm_layer.forward(z, DO_rate)
        q = z[:,-1:,:] if CPT==1 else z
        self.CPT = CPT
        c = self.attention_layer.forward(z, q)     # z:key&value, q:query
        y = np.concatenate((c, q), axis=-1)
        y = self.output_layer.forward(y)
        return y

    def backward(self, gy):
        gy = self.output_layer.backward(gy)
        B, T, H2 = gy.shape; H = H2//2
        gc = gy[:,:,:H]; gq = gy[:,:,H:]
        gz, gqa = self.attention_layer.backward(gc) # gz:key&valueの勾配、gq:queryの勾配
        gq += gqa
        if self.CPT==1:
            gz[:,-1:,:] += gq
        else:
            gz += gq
        gz = self.lstm_layer.backward(gz)
        gx = self.embedding_layer.backward(gz)
        return gx
        
