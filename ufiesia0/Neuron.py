from ufiesia0.Config import *
print(__file__)
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
            width = np.sqrt(1/m)  # Xavierの初期値
        self.w = (width * np.random.randn(m, n)).astype(Config.dtype) 
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
    def forward(self, x, r, c, w, v, b): 
        u = np.dot(x, w) + np.dot(r, v) + b
        y = np.tanh(u)
        self.state = x, r, y
        return y, c                    

    def backward(self, grad_y, grad_c, w, v):
        x, r, y = self.state
        delta = grad_y * (1 - y**2)  # tanhの逆伝播
        grad_w = np.dot(x.T, delta)
        grad_v = np.dot(r.T, delta)
        grad_b = np.sum(delta, axis=0)
        grad_x = np.dot(delta, w.T)
        grad_r = np.dot(delta, v.T)
        return grad_x, grad_r, grad_c, grad_w, grad_v, grad_b

class GRU_Unit:
    def forward(self, x, r, c, w, v, b):
        B, n = r.shape
        wg, wm = w[:, :2*n], w[:, 2*n:]
        vg, vm = v[:, :2*n], v[:, 2*n:]
        bg, bm = b[:2*n],    b[2*n:]
   
        # 更新ゲートとリセットゲート
        gu = np.dot(x, wg) + np.dot(r, vg) + bg
        g = 1 / (1 + np.exp(-gu))          # sigmoid
        gz, gr = g[:, :n], g[:, n:]        # 更新ゲートとリセットゲート
        # 新しい記憶
        u = np.dot(x, wm) + np.dot(gr * r, vm) + bm
        y = (1 - gz) * r + gz * np.tanh(u)
        self.state = x, r, y, g 
        return y, c

    def backward(self, grad_y, grad_c, w, v):  
        x, r, y, g = self.state
        B, n = r.shape
        wg, wm = w[:, :2*n], w[:, 2*n:]
        vg, vm = v[:, :2*n], v[:, 2*n:]
        gz, gr = g[:, :n],   g[:, n:]
        
        # y算出の逆伝播
        tanh_u  = (y - (1 - gz) * r) / gz
        grad_r  = grad_y * (1 - gz) 
        grad_gz = grad_y * (tanh_u - r) 
        
        # 新しい記憶　
        delta_m = grad_y * gz * (1 - tanh_u ** 2) 
        grad_wm = np.dot(x.T, delta_m)
        grad_vm = np.dot((gr * r).T, delta_m)
        grad_bm = np.sum(delta_m, axis=0)
        grad_x  = np.dot(delta_m, wm.T)
        grad_rm = np.dot(delta_m, vm.T)     # gr * r : リカレントな記憶の勾配
            
        # gr * r の逆伝播 
        grad_r += gr * grad_rm
        grad_gr = grad_rm * r 

        # 更新ゲートとリセットゲート
        delta_g = np.hstack((grad_gz, grad_gr)) * g * (1 - g) # sigmoidの微分
        grad_wg = np.dot(x.T, delta_g)
        grad_vg = np.dot(r.T, delta_g)
        grad_bg = np.sum(delta_g, axis=0)
        grad_w  = np.hstack((grad_wg, grad_wm))
        grad_v  = np.hstack((grad_vg, grad_vm))
        grad_b  = np.hstack((grad_bg, grad_bm))
        grad_x += np.dot(delta_g, wg.T)
        grad_r += np.dot(delta_g, vg.T)
        
        return grad_x, grad_r, grad_c, grad_w, grad_v, grad_b    

class LSTM_Unit:
    def forward(self, x, r, cp, w, v, b):  # 入力、前時刻状態
        B, n = r.shape
        u = np.dot(x, w) + np.dot(r, v) + b
        gz = 1 / (1 + np.exp(-u[:, :3*n])) # sigmoid 諸ゲート
        gm = np.tanh(u[:, 3*n:])           # tanh  新しい記憶
        gf = gz[:, :n]                     # 忘却ゲート
        gi = gz[:, n:2*n]                  # 入力ゲート
        go = gz[:, 2*n:]                   # 出力ゲート
        cn = cp * gf + gm * gi             # 旧記憶＊忘却ゲート＋新記憶＊入力ゲート
        y = np.tanh(cn) * go               # 記憶＊出力ゲート
        g = np.hstack((gz, gm))            # 内部状態
        self.state = x, r, cp, y, cn, g
        return y, cn

    def backward(self, grad_y, grad_cn, w, v):
        x, r, cp, y, cn, g = self.state
        B, n = r.shape
        gz = g[:,:3*n]                  # 忘却ゲート、入力ゲート、出力ゲート
        gf = g[:,:n]                    # 忘却ゲート
        gi = g[:,n:2*n]                 # 入力ゲート
        go = g[:,2*n:3*n]               # 出力ゲート
        gm = g[:,3*n:]                  # 新しい記憶
        tanh_c = np.tanh(cn)
        dcn = grad_cn + (grad_y * go) * (1 - tanh_c ** 2)
        dgm = dcn * gi                  # 新しい記憶の勾配

        # 諸ゲートの勾配： 忘却 dgf  入力 dgi  出力 dgo
        dgz = np.hstack((dcn * cp, dcn * gm, grad_y * tanh_c))

        # 諸ゲート sigmoidの微分 と 新しい記憶  tanhの微分
        delta = np.hstack((dgz * gz * (1 - gz), dgm * (1 - gm ** 2)))
            
        grad_cp = dcn * gf
        grad_w = np.dot(x.T, delta)
        grad_v = np.dot(r.T, delta)
        grad_b = np.sum(delta, axis=0)
        grad_x = np.dot(delta, w.T)
        grad_r = np.dot(delta, v.T)

        return grad_x, grad_r, grad_cp, grad_w, grad_v, grad_b 

# -- RNN層 -- 
class RnnBaseLayer:
    def __init__(self, *configuration, **kwargs):
        self.unit = None
        print('Initailize', self.__class__.__name__)
        self.width      = kwargs.pop('width',     None)
        optimizer_name  = kwargs.pop('optimize', 'SGD')

        self.w = None; self.v = None; self.b = None
        self.optimizer_w = cf.eval_in_module(optimizer_name, Optimizers)
        self.optimizer_v = cf.eval_in_module(optimizer_name, Optimizers)
        self.optimizer_b = cf.eval_in_module(optimizer_name, Optimizers) 

        self.config = configuration
        self.reset_state()  # 修正20230208
        #self.grad_r0, self.grad_c0 = None, None # 修正20230208

    def init_parameter(self, l, m, n): # l:戻りパス、m:入力、n:ニューロン数
        width = self.width
        if width is not None:
            width_w = float(width)
            width_v = float(width)
        else:                                
            width_w = np.sqrt(1/m) # Xavierの初期値
            width_v = np.sqrt(1/n) # Xavierの初期値
        self.w = (width_w * np.random.randn(m, n)).astype(Config.dtype)
        self.v = (width_v * np.random.randn(l, n)).astype(Config.dtype)
        self.b = np.zeros(n, dtype=Config.dtype)
        print(self.__class__.__name__, 'init_parameters', l, m, n)
        
    def forward(self, x):
        B, T, m = x.shape
        _, n = self.config
        # 時系列の展開の準備をして、リカレントにr0をセット    
        y = np.empty((B, T, n))
        self.layer = []    
        rt = np.zeros((B, n)) if self.r0 is None else self.r0
        ct = np.zeros((B, n)) if self.c0 is None else self.c0 # 追加20230208
        # unitを起こしながら、順伝播を繰返す    
        for t in range(T):
            unit = self.unit()
            xt = x[:, t, :]
            rt, ct = unit.forward(xt, rt, ct, self.w, self.v, self.b) # 修正20230208
            y[:, t, :] = rt
            self.layer.append(unit)
        self.r0, self.c0 = rt, ct # 修正20230208 last_y 
        return y
        
    def backward(self, grad_y):
        B, T, n = grad_y.shape
        m, _ = self.config
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)
        self.grad_x  = np.empty((B, T, m))
        grad_rt, grad_ct = 0, 0 # 修正20230208        
        for t in reversed(range(T)):
            unit = self.layer[t]
            grad_yt = grad_y[:, t, :] + grad_rt # 出力からとリカレントからの勾配を合算

            grad_xt, grad_rt, grad_ct, grad_wt, grad_vt, grad_bt = \
                        unit.backward(grad_yt, grad_ct, self.w, self.v) # grad_rt上書き
            # 上記修正20230208
            
            self.grad_w += grad_wt
            self.grad_v += grad_vt
            self.grad_b += grad_bt
            self.grad_x[:, t, :] = grad_xt
        self.grad_r0, self.grad_c0 = grad_rt, grad_ct # 修正20230208
        return self.grad_x

    def update(self, **kwargs):
        self.w -= self.optimizer_w.update(self.grad_w, **kwargs)
        self.v -= self.optimizer_v.update(self.grad_v, **kwargs)
        self.b -= self.optimizer_b.update(self.grad_b, **kwargs)
        
    def reset_state(self):
        self.r0, self.c0 = None, None # 修正20230208
        #self.grad_r0, self.grad_c0 = None, None # 修正20230208
        #self.layer = []

class RnnLayer(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        m, n = self.config
        super().init_parameter(n, m, n)
        self.unit = RNN_Unit

class GRU(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        m, n = self.config
        super().init_parameter(n, m, n*3)
        self.unit = GRU_Unit

class LSTM(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        m, n = self.config
        super().init_parameter(n, m, n*4)
        self.unit = LSTM_Unit

class RnnLayerForAgent(RnnLayer):
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
        # ユニットを変数とともにlayerに保存、次時刻に備えr0に出力をセット
        self.layer.append(unit)
        self.r0 = y         # last_y
        return y

    def get_stacked_states(self):
        m, n = self.config
        B = len(self.r0)
        T = len(self.layer)
        xs = np.empty((B, T, m))
        rs = np.empty((B, T, n))
        ys = np.empty((B, T, n))
        for t, unit in enumerate(self.layer): # 時刻毎に積まれたunitを取出す
            x, r, y = unit.state
            xs[:,t,:] = x
            rs[:,t,:] = r
            ys[:,t,:] = y
        #print(xs.shape, rs.shape, ys.shape)
        return xs, rs, ys
        
    def get_stacked_states2(self):
        xs, rs, ys = [], [], []
        for unit in self.layer: # 時刻毎に積まれたunitを取出す
            x, r, y = unit.state
            xs.append(x)
            rs.append(r)
            ys.append(y)
        xs = np.array(xs).transpose(1, 0, 2) # 時刻を軸１に　  
        rs = np.array(rs).transpose(1, 0, 2) # 時刻を軸１に
        ys = np.array(ys).transpose(1, 0, 2) # 時刻を軸１に
        #print(xs.shape, rs.shape, ys.shape)
        return xs, rs, ys
    

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
        #self.y = np.array([])
        self.optimizer_w = cf.eval_in_module(optimizer_name, Optimizers)
        self.init_parameter(m, n)

    def init_parameter(self, m, n):
        width = self.width
        if width is not None:
            width = float(width)  # 明示的に値を指定
        else:
            width = np.sqrt(1/n)  # 語ベクトルの各要素の活性化のため(通常のNeuronLayerとは違う)　 
        self.w = (width * np.random.randn(m, n)).astype(Config.dtype)
        print(self.__class__, 'init_parameters', m, n)

    def update(self, **kwargs):
        self.w -= self.optimizer_w.update(self.grad_w, **kwargs)

    def forward(self, x, DO_rate=0.0):
        self.x = x                # 入力 x は w のどの行を抽出するかを示す　
        #B, T = x.shape            # B:バッチサイズ、T:展開時間
        #m, n = self.config
        y = self.w[x]             # yはxの指すwの行を並べたもの
        return y                  # yの形状は(B, T, n)
                                  # 即ち長さnのベクトルがバッチ数×展開時間だけ並ぶ
    def backward(self, dy):
        #m, n    = self.config     
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        for i, idx in enumerate(self.x):
            self.grad_w[idx] += dy[i]
        #np.add.at(self.grad_w, self.x, dy)
        #np.scatter_add(self.grad_w, self.x, dy)

