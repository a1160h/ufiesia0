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
        print('Initialize', self.__class__, self.config)    
        self.width      = kwargs.pop('width',     None)
        optimizer_name  = kwargs.pop('optimize', 'SGD') 
        optimize_option = kwargs  # 残りは最適化のオプション
                
        self.w = None
        self.y = np.array([])
        self.optimizer_w = cf.eval_in_module(optimizer_name, Optimizers, **optimize_option)
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
            self.grad_w[idx] += self.dy[i]

        #np.add.at(self.grad_w, self.x, self.dy)
        
