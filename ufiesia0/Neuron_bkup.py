from ufiesia0.Config import *
from ufiesia0 import Activators, Optimizers
from ufiesia0 import common_function as cf

class BaseLayer: # ニューロンに共通の機能 
    def __init__(self, **kwargs):
        print('Initialize', self.__class__.__name__)
        self.width      = kwargs.pop('width',          None)
        activator_name  = kwargs.pop('activate', 'Identity') 
        optimizer_name  = kwargs.pop('optimize',      'SGD')  

        self.w = None; self.b = None
        self.activator = cf.eval_in_module(activator_name, Activators)
        self.optimizer_w = cf.eval_in_module(optimizer_name, Optimizers)
        self.optimizer_b = cf.eval_in_module(optimizer_name, Optimizers) 

    def init_parameter(self, m, n):
        if self.width is not None:
            width = self.width
        else:                                
            width = np.sqrt(1/m)  # # Xavierの初期値
        width = float(width)      # 精度の担保のため
        self.w = width * np.random.randn(m, n).astype(Config.dtype) 
        self.b = np.zeros(n, dtype=Config.dtype)                   
        print(self.__class__.__name__, 'init_parameters', m, n, Config.dtype)

    def init_parameter_bkup(self, m, n):
        width = self.width
        if width is not None:
            width = float(width)
        else:                                
            width = 0.01
        self.w = width * np.random.randn(m, n).astype(Config.dtype) 
        self.b = np.zeros(n, dtype=Config.dtype)                   
        print(self.__class__.__name__, 'init_parameters', m, n)

    def update(self, **kwargs):
        self.w -= self.optimizer_w.update(self.grad_w, **kwargs) # 戻り値=更新量
        self.b -= self.optimizer_b.update(self.grad_b, **kwargs) # 戻り値=更新量

class BaseLayer_bkup: # ニューロンに共通の機能 
    def __init__(self, **kwargs):
        print('Initialize', self.__class__.__name__)
        self.width      = kwargs.pop('width',          None)
        activator_name  = kwargs.pop('activate', 'Identity') 
        optimizer_name  = kwargs.pop('optimize',      'SGD')  

        self.w = None; self.b = None
        self.activator   = cf.eval_in_module(activator_name, Activators)
 
    def init_parameter(self, m, n):
        width = self.width
        if width is not None:
            width = float(width)
        else:                                
            width = 0.01
        self.w = width * np.random.randn(m, n).astype(Config.dtype) 
        self.b = np.zeros(n, dtype=Config.dtype)                   
        print(self.__class__.__name__, 'init_parameters', m, n)

    def update(self, eta=0.01, **kwargs):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

class NeuronLayer(BaseLayer): # 基本的なニューロンの機能
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.config = m, n
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        m = 1
        for i in shape[1:]:  # shape[0]はバッチサイズ
            m *= i
        self.config = m, self.config[1]
        print('NeuronLayer fix_configuration', shape, self.config)

    def forward(self, x):
        if None in self.config:
            print('NeuronLayer.input.shape', x.shape)
            self.fix_configuration(x.shape)
        m, n = self.config   # m:入力数、n:ニューロン数  
        if self.w is None or self.b is None:
            self.init_parameter(m, n)
        self.x = x.reshape(-1, m)
        u = np.dot(self.x, self.w) + self.b               # affine
        y_shape = x.shape[:-1] + (n,)
        u = u.reshape(y_shape)
        y = self.activator.forward(u)                     # 活性化関数
        return y

    def backward(self, grad_y, **kwargs):
        delta = self.activator.backward(grad_y, **kwargs) # 活性化関数
        m, n = self.config
        delta = delta.reshape(-1, n)
        self.grad_w = np.dot(self.x.T, delta)             # affine
        self.grad_b = np.sum(delta, axis=0)               # affine 
        grad_x = np.dot(delta, self.w.T)                  # affine
        x_shape = grad_y.shape[:-1] + (m,) 
        return grad_x.reshape(x_shape)

####
#
#
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

class SimpleRnnLayer:
    def __init__(self, *configuration, **kwargs):
        print('Initailize', self.__class__.__name__)
        self.width      = kwargs.pop('width',     None)
        optimizer_name  = kwargs.pop('optimize', 'SGD')  

        self.optimizer_w = cf.eval_in_module(optimizer_name, Optimizers)
        self.optimizer_v = cf.eval_in_module(optimizer_name, Optimizers)
        self.optimizer_b = cf.eval_in_module(optimizer_name, Optimizers) 
        
        self.w = None; self.v = None; self.b = None
        m, n = configuration
        self.config = m, n
        self.init_parameter(m, m, n) # x,rとも同形状
        self.unit = RNN_Unit()

    def init_parameter(self, l, m, n): # l:戻りパス、m:入力、n:ニューロン数
        width = self.width
        if width is not None:
            width_w = float(width)
            width_v = float(width)
        else:                                
            width_w = np.sqrt(1/m) # Xavierの初期値
            width_v = np.sqrt(1/l) # Xavierの初期値　
        self.w = width_w * np.random.randn(m, n).astype(Config.dtype)
        self.v = width_v * np.random.randn(l, n).astype(Config.dtype)
        self.b = np.zeros(n, dtype=Config.dtype)
        self.layers = []
        self.r0  = None
        print(self.__class__.__name__, 'init_parameters', l, m, n)
        
    def forward(self, x, r):
        y = self.unit.forward(x, r, self.w, self.v, self.b)
        return y
        
    def backward(self, grad_y):
        grad_x, grad_r, grad_w, grad_v, grad_b = \
                    self.unit.backward(grad_y, self.w, self.v)
        self.grad_w = grad_w
        self.grad_v = grad_v
        self.grad_b = grad_b
        return grad_x, grad_r

    def update(self, **kwargs):
        self.w -= self.optimizer_w.update(self.grad_w, **kwargs)
        self.v -= self.optimizer_v.update(self.grad_v, **kwargs)
        self.b -= self.optimizer_b.update(self.grad_b, **kwargs)

# -- RNN層 -- 
class RnnLayer:
    def __init__(self, *configuration, **kwargs):
        print('Initailize', self.__class__.__name__)
        self.width      = kwargs.pop('width',     None)
        optimizer_name  = kwargs.pop('optimize', 'SGD')  

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

class RNN(SimpleRnnLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
    

#### 時系列データをまとめて処理する Embedding層 #######################
# m:vocab_size(語彙数)、n:wordvec_size(語ベクトル長)
# w:その行に対応する語のベクトルを各行が示す(行数m=語彙数、列数n=語ベクトル長)
#   全体で単語の分散表現
class Embedding:
    def __init__(self, *configuration, **kwargs):
        self.name = 'embedding'
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = 10000; n, = configuration
        if len(configuration) == 0:
            m = 10000; n = 100
        self.config = m, n    
        print('Initialize', self.__class__.__name__, self.config)    
        self.width      = kwargs.pop('width',     None)
        optimizer_name  = kwargs.pop('optimize', 'SGD') 
        optimize_option = kwargs  # 残りは最適化のオプション
                
        self.w = None
        self.y = np.array([])
        self.optimizer_w = cf.eval_in_module(optimizer_name, Optimizers)
        self.init_parameter(m, n)

    def init_parameter(self, m, n):
        width = self.width
        if width is not None:
            width = float(width)  # 明示的に値を指定
        else:
            width = np.sqrt(1/n)  # 語ベクトルの各要素の活性化のため(通常のNeuronLayerとは違う)　 
        self.w = width * np.random.randn(m, n).astype(Config.dtype)
        print(self.__class__, 'init_parameters', m, n)

    def update(self, **kwargs):
        self.w -= self.optimizer_w.update(self.grad_w, **kwargs)

    def forward(self, x, DO_rate=0.0):
        self.x = x                # 入力 x は w のどの行を抽出するかを示す　
        B, T = x.shape            # B:バッチサイズ、T:展開時間
        m, n = self.config
        y = self.w[x, :]          # yはxの指すwの行を並べたもの
        return y                  # yの形状は(B, T, n)
                                  # 即ち長さnのベクトルがバッチ数×展開時間だけ並ぶ
    def backward(self, dy):
        m, n    = self.config     
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        for i, idx in enumerate(self.x):
            self.grad_w[idx] += dy[i]
        #np.add.at(self.grad_w, self.x, dy)
        
