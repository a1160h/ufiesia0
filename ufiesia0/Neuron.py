from ufiesia0.Config import *
print(__file__)
from ufiesia0 import Activators, Optimizers
from ufiesia0 import common_function as cf

class BaseLayer:
    """ ニューロンに共通の機能 """
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


class NeuronLayer(BaseLayer):
    """ 基本的なニューロンの機能 """
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.config = m, n
        self.full_cnnt = kwargs.pop('full_connection', False) # 全結合層を明示
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        m = 1
        for i in shape[1:]:  # shape[0]はバッチサイズ
            m *= i
        self.config = m, self.config[1]
        print('NeuronLayer fix_configuration', shape, self.config)

    def forward(self, x):
        if None in self.config:
            print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        m, n = self.config   # m:入力数、n:ニューロン数  
        if self.w is None or self.b is None:
            self.init_parameter(m, n)
            
        self.x = x.reshape(-1, m)
        u = np.dot(self.x, self.w) + self.b               # affine
        y = self.activator.forward(u)                     # 活性化関数
        y_shape = (-1, n) if self.full_cnnt is True else (-1,) + x.shape[1:-1] + (n,) 
        return y.reshape(*y_shape)

    def backward(self, grad_y, **kwargs):
        m, n = self.config
        grad_y = grad_y.reshape(-1, n)
        delta = self.activator.backward(grad_y, **kwargs) # 活性化関数
        #delta = delta.reshape(-1, n)
        self.grad_w = np.dot(self.x.T, delta)             # affine
        self.grad_b = np.sum(delta, axis=0)               # affine 
        grad_x = np.dot(delta, self.w.T)                  # affine
        x_shape = grad_y.shape[:-1] + (m,) 
        return grad_x.reshape(x_shape)

### 畳み込み層 #####################################################
class PremitiveConvLayer(BaseLayer):
    """ 畳み込み層(動作原理に忠実な基本版) """
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高, Fw:フィルタ幅
    # Sh:ストライド高，Sw:ストライド幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 6:
            C, image_size, M, kernel_size, stride, pad = configuration
        elif len(configuration) == 4:
            C = None; image_size = None
            M, kernel_size, stride, pad = configuration
        elif len(configuration) == 2:
            C = None; image_size = None
            M, kernel_size = configuration
            stride = 1; pad = 0 
        else:
            raise Exception('cannot initialize ' + self.__class__.__name__)   
        Oh = None; Ow = None

        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) \
                            else (image_size, image_size)
        Fh, Fw = kernel_size if isinstance(kernel_size, (tuple, list)) \
                             else (kernel_size, kernel_size)
        Sh, Sw = stride if isinstance(stride, (tuple, list)) \
                        else (stride, stride)
        
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        self.grad_w, self.grad_b = None, None
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if (C is None or Ih is None or Iw is None) and len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
            
        Oh = (Ih - Fh + 2*pad) // Sh + 1   # 出力高さ
        Ow = (Iw - Fw + 2*pad) // Sw + 1   # 出力幅
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        
    def forward(self, x):
        if None in self.config:
            self.fix_configuration(x.shape)
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if self.w is None:
            self.init_parameter(C*Fh*Fw, M)
          
        B = x.size // (C*Ih*Iw) # B = x.shape[0] = len(x)
        # 画像調整  (C,Ih*Iw)にも対応            B     C     Ih上Ih下  Iw左Iw右　ゼロパディング   
        self.x = np.pad(x.reshape(B,C,Ih,Iw), [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant')
        
        u = np.zeros((B,Oh,Ow,M), dtype=Config.dtype)
        for ih in range(0, Ih-Fh+2*pad+1, Sh):       # ih+FhがIhからはみ出さないように
            for iw in range(0, Iw-Fw+2*pad+1, Sw):   # iw+FwがIwからはみ出さないように 
                xij = self.x[:,:,ih:ih+Fh, iw:iw+Fw] # xのFh*Fwの領域を取出す
                xij = xij.reshape(B, -1)
                uij = np.dot(xij, self.w) + self.b   # 取出した領域を共通のwとbでaffine変換
                u[:,ih//Sh,iw//Sw,:] = uij           # uの該当箇所に値を設定
        u = u.transpose(0,3,1,2)
        y = self.activator.forward(u)                      
        return y
    
    def backward(self, grad_y):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        B = grad_y.size // (M*Oh*Ow)
        self.grad_y = grad_y.reshape(B, M, Oh, Ow)
        delta = self.activator.backward(self.grad_y)       
        delta = delta.transpose(0,2,3,1) 

        gx = np.zeros_like(self.x, dtype=Config.dtype)
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        self.grad_b = np.zeros_like(self.b, dtype=Config.dtype)

        for ih in range(0, Ih-Fh+2*pad+1, Sh):
            for iw in range(0, Iw-Fw+2*pad+1, Sw):
                xij = self.x[:,:,ih:ih+Fh, iw:iw+Fw]
                xij = xij.reshape(B, -1)
                guij = delta[:,ih//Sh,iw//Sw,:]
                gwij = np.dot(xij.T, guij)
                gbij = np.sum(guij, axis=0)
                gxij = np.dot(guij, self.w.T)
                gx[:,:,ih:ih+Fh, iw:iw+Fw] = gxij.reshape(B,C,Fh,Fw)
                self.grad_w += gwij
                self.grad_b += gbij
         
        # 順伝播で画像調整した分を戻す
        gx = gx[:, :, pad:pad+Ih, pad:pad+Iw]
        return gx

class ConvLayer(BaseLayer):
    """ 畳み込み層(img2col変換による高速版 numpyではあまり差がないがcupyで顕著) """
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高, Fw:フィルタ幅
    # Sh:ストライド高，Sw:ストライド幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 6:
            C, image_size, M, kernel_size, stride, pad = configuration
        elif len(configuration) == 4:
            C = None; image_size = None; M, kernel_size, stride, pad = configuration
        elif len(configuration) == 2:
            C = None; image_size = None; M, kernel_size = configuration; stride = 1; pad = 0 
        else:
            raise Exception('cannot initialize ' + self.__class__.__name__)   
        Oh = None; Ow = None

        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        Fh, Fw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        Sh, Sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if (C is None or Ih is None or Iw is None) and len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
            
        Oh = (Ih - Fh + 2*pad) // Sh + 1   # 出力高さ
        Ow = (Iw - Fw + 2*pad) // Sw + 1   # 出力幅
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        
    def forward(self, x):
        if None in self.config:
            self.fix_configuration(x.shape)
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if self.w is None:
            self.init_parameter(C*Fh*Fw, M)
          
        B = x.size // (C*Ih*Iw) # B = x.shape[0] = len(x)
        # 画像調整  (C,Ih*Iw)にも対応            B      C      Ih上 Ih下   Iw左 Iw右　 ゼロパディング   
        img_pad = np.pad(x.reshape(B,C,Ih,Iw), [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        cols = np.empty((B, C, Fh, Fw, Oh, Ow), dtype=Config.dtype)  # メモリ節約のためzerosでなくempty 
        # 入力画像とフィルタを行列に変換
        # img_pad.shape=(B,C,Ih+2*pad,Iw+2*pad) → cols.shape=(B,C,Fh,Fw,Oh,Ow)
        # img_padからstride毎のデータを取ってきて、colsにOh,Owになるまで並べる
        # それをFh,Fwを満たすまで繰返す
        for h in range(Fh):
            h_lim = h + Sh*Oh
            for w in range(Fw):
                w_lim = w + Sw*Ow
                cols[:, :, h, w, :, :] = img_pad[:, :, h:h_lim:Sh, w:w_lim:Sw]

        # 軸の入替と変形           B  Oh Ow C  Fh Fw
        self.cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(B*Oh*Ow, C*Fh*Fw)

        # 出力の計算: 行列積、バイアスの加算、活性化関数
        # C*Fh*Fwの掛算=フィルタによる畳み込み、このとき他(B,M)は同時に扱うが演算しない
        u = np.dot(self.cols, self.w) +self.b
        #print(u.shape)
        u = u.reshape(B, Oh, Ow, M).transpose(0, 3, 1, 2)
        y = self.activator.forward(u)                 
        return y
    
    def backward(self, grad_y):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        B = grad_y.size // (M*Oh*Ow)
        self.grad_y = grad_y.reshape(B, M, Oh, Ow)
        delta = self.activator.backward(self.grad_y)       
        delta = delta.transpose(0, 2, 3, 1).reshape(B*Oh*Ow, M)

        # フィルタとバイアスの勾配                   
        self.grad_w = np.dot(self.cols.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        grad_cols = np.dot(delta, self.w.T)

        # 入力の勾配 (B*Oh*Ow,M)×(M,C*Fh*Fw)
        #                                                   B  C  Fh Fw Oh Ow
        cols = grad_cols.reshape(B,Oh,Ow,C,Fh,Fw).transpose(0, 3, 4, 5, 1, 2)

        images = np.zeros((B, C, Ih+2*pad, Iw+2*pad), dtype=Config.dtype)
        # 行列を入力画像に逆変換
        # cols.shape=(B,C,Fh,Fw,Oh,Ow) → img_pad.shape=(B,C,Ih+2*pad,Iw+2*pad) 
        # colsからstride*Oh,Ow個のデータを取ってきて、img_padにstride毎に並べる
        # それをFh,Fwを満たすまで繰返す
        for h in range(Fh):
            h_lim = h + Sh*Oh
            for w in range(Fw):
                w_lim = w + Sw*Ow
                images[:, :, h:h_lim:Sh, w:w_lim:Sw] += cols[:, :, h, w, :, :]

        # 順伝播で画像調整した分を戻す
        grad_x = images[:, :, pad:pad+Ih, pad:pad+Iw]

        return grad_x

### プーリング層 ####################################################
class PoolingLayer:
    """ Max Poolingによるpooling層 """
    # strideはpool長と同じで重複無しで、右・下端の端数は切捨てる 
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Oh:出力高さ, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        method = None
        if   len(configuration) == 5:
            C, image_size, pool, pad, method = configuration
        elif len(configuration) == 4:
            C, image_size, pool, pad = configuration
        elif len(configuration) == 3 and isinstance(configuration[-1], str):
            C = None; image_size = None; pool, pad, method = configuration
        elif len(configuration) == 3:
            C, image_size, pool = configuration; pad = 0
        elif len(configuration) == 2 and isinstance(configuration[-1], str):
            C = None; image_size = None; pool, method = configuration; pad = 0
        elif len(configuration) == 2:    
            C = None; image_size = None; pool, pad = configuration
        elif len(configuration) == 1:
            C = None; image_size = None; pool = configuration; pad = 0
        else:
            C = None; image_size = None; pool = 2; pad = 0

        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        pool_h, pool_w = pool if isinstance(pool, (tuple, list)) else (pool, pool)
        Oh = Ow = None
        self.max_index = None
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
        print('Initialize', self.__class__.__name__, self.config)
        
    def fix_configuration(self, shape):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        B  = shape[0]
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__, 'cannot fix configuration.')

        Oh = (Ih + 2 * pad) // pool_h # 端数は切捨て
        Ow = (Iw + 2 * pad) // pool_w # 端数は切捨て
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
            
    def forward(self, x):
        if None in self.config:
            self.fix_configuration(x.shape)
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config

        B = x.size // (C*Ih*Iw)                   # B = x.shape[0] = len(x)
        x = x.reshape(B, C, Ih, Iw)               # 入力の形状 ex. (C,Ih*Iw)に対応   

        # 画像調整            B      C      Ih上 Ih下   Iw左 Iw右　 ゼロパディング   
        img_pad = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        img_pad = img_pad[..., :Oh*pool_h, :Ow*pool_w] # 端数の切り捨て

        # Ih+pad+pdh=Oh*pool_h; Iw+pad+pdw=Ow*pool_w      軸をpoolで分割できる
        # 入力画像を変形（B,C, Ih+pad+pdh, Iw+pad+pdw) → (B,C,Oh,Ow,Pool_h*Pool_w)
                              #   (0, 1, 2,  3,      4,   5   )
        quarry = img_pad.reshape  (B, C, Oh, pool_h, Ow,  pool_w)   \
                        .transpose(0, 1, 2,  4,      3,   5)        \
                        .reshape  (B, C, Oh, Ow,     pool_h*pool_w)
                              #   (0, 1, 2,  3,      4)  pool_h*pool_wはaxis=4 

        # 出力の計算
        y = np.max (quarry, axis=4)   # pool*poolの軸で最大値
        self.max_index = np.argmax(quarry, axis=4)  # インデクス記録
            
        return y

    def backward(self, grad_y):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config   
        B = grad_y.size // (C*Oh*Ow)                 # B = grad_y.shape[0] = len(grad_y)
        self.grad_y = grad_y.reshape(B, C, Oh, Ow)   # ドロップアウトへの入力形状は順伝播時と同じ
        grad_cols = np.zeros((B*C*Oh*Ow,pool_h*pool_w), dtype=Config.dtype)  # 初期値0
        
        # 各行の最大値であった列の要素にのみ出力の勾配を入れる
        max_index = self.max_index.reshape(-1)
        grad_cols[np.arange(B*C*Oh*Ow), max_index] = self.grad_y.reshape(-1)

                              #     (B*C*Oh*Ow,      pool_h*pool_w)
        grad_x = grad_cols.reshape  (B*C, Oh,Ow,     pool_h,pool_w) \
                          .transpose(0,   1, 3,      2,     4)      \
                          .reshape  (B,C, Oh*pool_h, Ow*pool_w) 
                              #     (B,C, Ih+pad+pdh,Iw+pad+pdw) 

        # 順伝播で画像調整した分を戻す
        trancated_h = Ih + 2*pad - Oh*pool_h
        trancated_w = Iw + 2*pad - Ow*pool_w
        grad_x = np.pad(grad_x, [(0,0),(0,0),(0,trancated_h),(0,trancated_w)], 'constant')
        grad_x = grad_x[:, :, pad:pad+Ih, pad:pad+Iw]  # grad_x.shape=(B,C,Ih,Iw) 
        return grad_x

####
#
#
class RNN_Unit:
    """ Recurrent Neural Network Unit：基本のRNN機能(ユニット部分) """
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
    """ Gated Recurrent Unit：ゲートを備え高度化したRNNの進化系の1種 """
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
    """ Long Short Time Memory Unit：ゲートを備え高度化したRNNの進化形 """
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
        gz = g[:,:3*n]                     # 忘却ゲート、入力ゲート、出力ゲート
        gf = g[:,:n]                       # 忘却ゲート
        gi = g[:,n:2*n]                    # 入力ゲート
        go = g[:,2*n:3*n]                  # 出力ゲート
        gm = g[:,3*n:]                     # 新しい記憶
        tanh_c = np.tanh(cn)
        dcn = grad_cn + (grad_y * go) * (1 - tanh_c ** 2)
        dgm = dcn * gi                     # 新しい記憶の勾配

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
    """ RNNを層として構成するためのベース """
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
        self.reset_state()  

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
        ct = np.zeros((B, n)) if self.c0 is None else self.c0 
        # unitを起こしながら、順伝播を繰返す    
        for t in range(T):
            unit = self.unit()
            xt = x[:, t, :]
            rt, ct = unit.forward(xt, rt, ct, self.w, self.v, self.b) 
            y[:, t, :] = rt
            self.layer.append(unit)
        self.r0, self.c0 = rt, ct # last_y 
        return y
        
    def backward(self, grad_y):
        B, T, n = grad_y.shape
        m, _ = self.config
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)
        self.grad_x  = np.empty((B, T, m))
        grad_rt, grad_ct = 0, 0       
        for t in reversed(range(T)):
            unit = self.layer[t]
            grad_yt = grad_y[:, t, :] + grad_rt # 出力からとリカレントからの勾配を合算

            grad_xt, grad_rt, grad_ct, grad_wt, grad_vt, grad_bt = \
                        unit.backward(grad_yt, grad_ct, self.w, self.v) # grad_rt上書き
            
            self.grad_w += grad_wt
            self.grad_v += grad_vt
            self.grad_b += grad_bt
            self.grad_x[:, t, :] = grad_xt
        self.grad_r0, self.grad_c0 = grad_rt, grad_ct 
        return self.grad_x

    def update(self, **kwargs):
        self.w -= self.optimizer_w.update(self.grad_w, **kwargs)
        self.v -= self.optimizer_v.update(self.grad_v, **kwargs)
        self.b -= self.optimizer_b.update(self.grad_b, **kwargs)
        
    def reset_state(self):
        self.r0, self.c0 = None, None 

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
        return xs, rs, ys
    

class Embedding:
    """ 時系列データをまとめて処理する Embedding層 """
    # m:vocab_size(語彙数)、n:wordvec_size(語ベクトル長)
    # w:その行に対応する語のベクトルを各行が示す(行数m=語彙数、列数n=語ベクトル長)
    #   全体で単語の分散表現
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
        y = self.w[x]             # yはxの指すwの行を並べたもの
        return y                  # yの形状は(B, T, n)
                                  # 即ち長さnのベクトルがバッチ数×展開時間だけ並ぶ
    def backward(self, dy):
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        for i, idx in enumerate(self.x):
            self.grad_w[idx] += dy[i]
        #np.add.at(self.grad_w, self.x, dy)
        #np.scatter_add(self.grad_w, self.x, dy)

