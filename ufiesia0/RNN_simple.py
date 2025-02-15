### RNN_simple ごく基本のRNN
# 2022.09.08 A.Inoue

from ufiesia0.Config import *
import pickle
from ufiesia0 import Neuron, LossFunctions, Activators, Optimizers
from ufiesia0 import common_function as cf

class RNN_Unit:
    def forward(self, x, r, w, v, b): 
        u = np.dot(x, w) + np.dot(r, v) + b
        y = np.tanh(u)
        self.state = x, r, y
        return y                    

    def backward(self, grad_y, w, v):
        x, r, y = self.state
        delta = grad_y * (1 - y**2)  # tanhの逆伝播
        grad_w = np.dot(x.T, delta)
        grad_v = np.dot(r.T, delta)
        grad_b = np.sum(delta, axis=0)
        grad_x = np.dot(delta, w.T)
        grad_r = np.dot(delta, v.T)
        return grad_x, grad_r, grad_w, grad_v, grad_b

# -- RNN層 -- 
class SimpleRnnLayer:
    def __init__(self, *configuration, **kwargs):
        print('Initailize', self.__class__.__name__)
        self.width      = kwargs.pop('width',          None)
        optimizer_name  = kwargs.pop('optimize',      'SGD')  

        self.optimizer_w = cf.eval_in_module(optimizer_name, Optimizers)
        self.optimizer_v = cf.eval_in_module(optimizer_name, Optimizers)
        self.optimizer_b = cf.eval_in_module(optimizer_name, Optimizers) 
        
        self.w = None; self.v = None; self.b = None
        m, n = configuration
        self.config = m, n
        self.init_parameter(n, m, n)

    def init_parameter(self, l, m, n): # l:戻りパス、m:入力、n:ニューロン数
        width = self.width
        if width is not None:
            width_w = float(width)
            width_v = float(width)
        else:                                
            width_w = np.sqrt(1/m) # Xavierの初期値
            width_v = np.sqrt(1/n) # Xavierの初期値　
        self.w = width_w * np.random.randn(m, n).astype(Config.dtype)
        self.v = width_v * np.random.randn(l, n).astype(Config.dtype)
        self.b = np.zeros(n, dtype=Config.dtype)
        self.layers = []
        self.r0  = None
        print(self.__class__.__name__, 'init_parameters', l, m, n)
        
    def forward(self, x):
        B, T, m = x.shape
        _, n = self.config
        # 時系列の展開の準備をして、リカレントにr0をセット    
        y = np.empty((B, T, n))
        self.layers = []    
        rt = np.zeros((B, n)) if self.r0 is None else self.r0
        # unitを起こしながら、順伝播を繰返す    
        for t in range(T):
            unit = RNN_Unit()
            xt = x[:, t, :]
            rt = unit.forward(xt, rt, self.w, self.v, self.b) # rt上書き
            y[:, t, :] = rt
            self.layers.append(unit)
        self.r0 = rt # last_y 
        return y
        
    def backward(self, grad_y):
        B, T, n = grad_y.shape
        m, _ = self.config
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)
        self.grad_x  = np.empty((B, T, m))
        grad_rt = 0
        
        for t in reversed(range(T)):
            unit = self.layers[t]
            grad_yt = grad_y[:, t, :] + grad_rt # 出力からとリカレントからの勾配を合算

            grad_xt, grad_rt, grad_wt, grad_vt, grad_bt = \
                        unit.backward(grad_yt, self.w, self.v) # grad_rt上書き

            self.grad_w += grad_wt
            self.grad_v += grad_vt
            self.grad_b += grad_bt
            self.grad_x[:, t, :] = grad_xt
        self.grad_r0 = grad_rt 
        return self.grad_x

    def update(self, **kwargs):
        self.w -= self.optimizer_w.update(self.grad_w, **kwargs)
        self.v -= self.optimizer_v.update(self.grad_v, **kwargs)
        self.b -= self.optimizer_b.update(self.grad_b, **kwargs)

    def step_only(self, x):
        """ １時刻ずつデータを処理、状態は変えない """
        m, n = self.config  # B, m = x.shape; B, n = t.shape
        if x.ndim==2:
            B, _ = x.shape
        else:
            B = 1
            x = x.reshape(B, m)
        if self.r0 is None: # 時系列の最初
            r = np.zeros((B, n)) 
        else:
            r = self.r0   
        y = RNN_Unit().forward(x, r, self.w, self.v, self.b)
        return y

    def step_and_stack(self, x):
        """ １時刻ずつデータを処理して状態を蓄積 """
        m, n = self.config  # B, m = x.shape; B, n = t.shape
        if x.ndim==2:
            B, _ = x.shape
        else:
            B = 1
            x = x.reshape(B, m)
        # リカレントにr0をセットし、unitを起こして順伝播    
        r = np.zeros((B, n)) if self.r0 is None else self.r0  
        unit = RNN_Unit()
        y = unit.forward(x, r, self.w, self.v, self.b)
        # ユニットを変数とともにlayersに保存、次時刻に備えr0に出力をセット
        self.layers.append(unit)
        self.r0 = y         # last_y
        return y

    def get_stacked_states(self):
        m, n = self.config
        B = len(self.r0)
        T = len(self.layers)
        xs = np.empty((B, T, m))
        rs = np.empty((B, T, n))
        ys = np.empty((B, T, n))
        for t, unit in enumerate(self.layers): # 時刻毎に積まれたunitを取出す
            x, r, y = unit.state
            xs[:,t,:] = x
            rs[:,t,:] = r
            ys[:,t,:] = y
        #print(xs.shape, rs.shape, ys.shape)
        return xs, rs, ys
        
    def get_stacked_states2(self):
        xs, rs, ys = [], [], []
        for unit in self.layers: # 時刻毎に積まれたunitを取出す
            x, r, y = unit.state
            xs.append(x)
            rs.append(r)
            ys.append(y)
        xs = np.array(xs).transpose(1, 0, 2) # 時刻を軸１に　  
        rs = np.array(rs).transpose(1, 0, 2) # 時刻を軸１に
        ys = np.array(ys).transpose(1, 0, 2) # 時刻を軸１に
        #print(xs.shape, rs.shape, ys.shape)
        return xs, rs, ys
        
    def reset_state(self):
        self.r0 = None
        self.layers = []

class SimpleRNN:
    def __init__(self, l, m, n, **kwargs):
        self.config = l, m, n
        self.rnn_layer = SimpleRnnLayer(l, m)
        self.neuron_layer = Neuron.NeuronLayer(m, n, **kwargs)
        loss_f = kwargs.pop('loss_function', 'MeanSquaredError')
        self.loss_function = cf.eval_in_module(loss_f, LossFunctions)
        self.gys = []
        self.loss = []
        self.layers = [self.rnn_layer, self.neuron_layer]

    def summary(self):
        print('～～ model summary of', self.__class__.__name__, '～'*22)
        for i, layer in enumerate(self.layers):
            print('layer', i, layer.__class__.__name__)
            print(' configuration =', layer.config, end='')
            if hasattr(layer, 'activator'):
                print('\n activate =', layer.activator.__class__.__name__, end=' ')
            print('\n' + '-'*72)
        if hasattr(self, 'loss_function'):
            print('loss_function =', self.loss_function.__class__.__name__)
        print('～～ end of summary ' + '～'*28 + '\n')

    def update(self, eta, g_clip=None):
        self.rnn_layer.update(eta=eta, g_clip=g_clip)
        self.neuron_layer.update(eta=eta, g_clip=g_clip)

    def reset_state(self):
        self.rnn_layer.reset_state()
        self.gys = []
        self.loss = []

    def forward(self, x):
        l, m, n = self.config
        B, T, _ = x.shape
        y = self.rnn_layer.forward(x)
        y = y.reshape(-1, m)
        y = self.neuron_layer.forward(y)
        y = y.reshape(B, T, n)
        return y

    def backward(self, grad_y):
        l, m, n = self.config
        B, T, n = grad_y.shape
        grad_x = grad_y.reshape(-1, n)
        grad_x = self.neuron_layer.backward(grad_x)
        grad_x = grad_x.reshape(B, T, m)
        grad_x = self.rnn_layer.backward(grad_x)
        return grad_x

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
        self.update(eta, g_clip)
        return np.mean(loss)
        
    def generate(self, seed, length=200, *args):
        """ seedに続いて一つずつ生成していく """
        T, l = seed.shape
        gen_data = np.zeros((length, l))
        gen_data[:T] = seed
        self.reset_state()
        for j in range(length-1):
            x = gen_data[j]
            y = self.forward(x.reshape(1, 1, l))
            if j+1 < T: # seedの範囲は書込まない
                continue
            gen_data[j+1] = y.reshape(-1)
        return gen_data

    def generate2(self, seed, length=200, verbose=False, extension=False):
        """
        extension==True:末尾に一つ追加しながら頭は変えずに伸ばしていく
        extension==False:末尾に一つ追加して頭を一つ後ろにずらす

        """
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

    def save_parameters(self, file_name):
        """ 学習結果の保存(辞書形式) """
        params = {}
        params['title'] = 'SimpleRNN'
        params['wr'] = np.array(self.rnn_layer.w)
        params['vr'] = np.array(self.rnn_layer.v)
        params['br'] = np.array(self.rnn_layer.b)
        params['wn'] = np.array(self.neuron_layer.w)
        params['bn'] = np.array(self.neuron_layer.b)
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print(self.__class__.__name__, 'モデルのパラメータをファイルに記録しました=>', file_name)    

    def load_parameters(self, file_name):
        """ 学習結果の継承(辞書形式) """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        title = params.pop('title', None)
        print(title, 'モデルのパラメータをファイルから取得しました<=', file_name)
        self.rnn_layer.w = np.array(params['wr'].tolist())
        self.rnn_layer.v = np.array(params['vr'].tolist())
        self.rnn_layer.b = np.array(params['br'].tolist())
        self.neuron_layer.w = np.array(params['wn'].tolist())
        self.neuron_layer.b = np.array(params['bn'].tolist())
        return title, params

     
