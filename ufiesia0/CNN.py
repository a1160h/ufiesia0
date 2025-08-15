from ufiesia0.Config import *
from ufiesia0 import NN, Neuron
from ufiesia0 import common_function as cf

class CNN_c(NN.NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        '''
        ニューラルネットワーク
        　畳込み層＋全結合層（出力層）
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer0 畳込み層
        M      = kwargs.pop('M',     24)            # フィルタ数
        kernel_size = kwargs.pop('kernel_size', 3)  # フィルタ高
        stride = kwargs.pop('stride', 1)            # ストライド
        cl_pad = kwargs.pop('cl_pad', 0)            # パディング
        opt_for_cl = {}
        opt_for_cl['activate']  = kwargs.pop('cl_act', 'ReLU')    # 活性化関数 
        # layer1 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['full_connection'] = True
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.conv_layer    = Neuron.ConvLayer(M, kernel_size, stride, cl_pad, **opt_for_cl)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer)
        self.layers.append(self.output_layer)

class CNN_cp(NN.NN_CNN_Base): #ニューラルネットワーク　畳込み層＋プーリング層＋全結合層（中間層＋出力層）
    def __init__(self, *args, **kwargs):
        '''
        ニューラルネットワーク
          畳込み層＋プーリング層
        ＋全結合層(出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer0 畳込み層
        M      = kwargs.pop('M',     24)            # フィルタ数
        kernel_size = kwargs.pop('kernel_size', 3)  # フィルタ高
        stride = kwargs.pop('stride', 1)            # ストライド
        cl_pad = kwargs.pop('cl_pad', 0)            # パディング
        opt_for_cl = {}
        opt_for_cl['activate']  = kwargs.pop('cl_act', 'ReLU')    # 活性化関数 
        # layer1 プーリング層
        opt_for_pl = {}
        pool   = kwargs.pop('pool',   2)            # プーリング
        pl_pad = kwargs.pop('pl_pad', 0)            # パディング
        # layer2 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        opt_for_ol['full_connection'] = True         
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl.update(kwargs) 
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.conv_layer    = Neuron.ConvLayer(M, kernel_size, stride, cl_pad, **opt_for_cl)
        self.pooling_layer = Neuron.PoolingLayer(pool, pl_pad, **opt_for_pl)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.output_layer)

class CNN_cpm(NN.NN_CNN_Base): #ニューラルネットワーク　畳込み層＋プーリング層＋全結合層（中間層＋出力層）
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層＋プーリング層
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer0 畳込み層
        M      = kwargs.pop('M',     24)            # フィルタ数
        kernel_size = kwargs.pop('kernel_size', 3)  # フィルタ高
        stride = kwargs.pop('stride', 1)            # ストライド
        cl_pad = kwargs.pop('cl_pad', 0)            # パディング
        opt_for_cl = {}
        opt_for_cl['activate']  = kwargs.pop('cl_act', 'ReLU')    # 活性化関数 
        # layer1 プーリング層
        opt_for_pl = {}
        pool   = kwargs.pop('pool',   2)            # プーリング
        pl_pad = kwargs.pop('pl_pad', 0)            # パディング
        # layer2 全結合中間層 
        ml_nn  = kwargs.pop('ml_nn',  200)          # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']  = kwargs.pop('ml_act', 'ReLU')    # 活性化関数
        opt_for_ml['full_connection'] = True         
        # layer3 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']  = kwargs.pop('ol_act', ol_act_cand) # 活性化関数
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl.update(kwargs) 
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.conv_layer    = Neuron.ConvLayer(M, kernel_size, stride, cl_pad, **opt_for_cl)
        self.pooling_layer = Neuron.PoolingLayer(pool, pl_pad, **opt_for_pl)
        self.middle_layer  = Neuron.NeuronLayer(ml_nn, **opt_for_ml)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)

class CNN_ccpm(NN.NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        '''
        入力の大きさは最初のfowardで決まる
        ニューラルネットワーク
          畳込み層×２＋プーリング層
        ＋全結合層(中間層＋出力層)
        '''
        # ニューラルネットワークの名称
        In, Out = super().__init__(*args, **kwargs)
        # layer0 畳込み層1
        M1      = kwargs.pop('M1',      24)          # フィルタ数
        kernel_size1 = kwargs.pop('kernel_size1', 2) # フィルタ高
        stride1 = kwargs.pop('stride1',  1)          # ストライド
        cl1_pad = kwargs.pop('cl1_pad',  0)          # パディング
        opt_for_cl1 = {}
        opt_for_cl1['activate']  = kwargs.pop('cl1_act', 'ReLU')    # 活性化関数 
        # layer1 畳込み層2
        M2      = kwargs.pop('M2',      24)          # フィルタ数
        kernel_size2 = kwargs.pop('kernel_size2', 2) # フィルタ高
        stride2 = kwargs.pop('stride2',  1)          # ストライド
        cl2_pad = kwargs.pop('cl2_pad',  0)          # パディング
        opt_for_cl2 = {}
        opt_for_cl2['activate']  = kwargs.pop('cl2_act', 'ReLU')    # 活性化関数 
        # layer2 プーリング層
        opt_for_pl = {}
        pool   = kwargs.pop('pool',      2)          # プーリング
        pl_pad = kwargs.pop('pl_pad',    0)          # パディング
        # layer3 全結合中間層 
        ml_nn  = kwargs.pop('ml_nn',  200)           # ニューロン数
        opt_for_ml = {}
        opt_for_ml['activate']   = kwargs.pop('ml_act', 'ReLU')     # 活性化関数
        opt_for_ml['full_connection'] = True         
        # layer4 全結合出力層　　　　
        opt_for_ol = {}
        ol_act_cand = 'identity' if Out == 1 else 'Softmax'
        opt_for_ol['activate']   = kwargs.pop('ol_act', ol_act_cand)# 活性化関数
        # kwargs に残ったものを結合(w_decay や最適化オプションなど)
        opt_for_cl1.update(kwargs) 
        opt_for_cl2.update(kwargs) 
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.conv_layer1   = Neuron.ConvLayer(M1, kernel_size1, stride1, cl1_pad, **opt_for_cl1)
        self.conv_layer2   = Neuron.ConvLayer(M2, kernel_size2, stride2, cl2_pad, **opt_for_cl2)
        self.pooling_layer = Neuron.PoolingLayer(pool, pl_pad, **opt_for_pl)
        self.middle_layer  = Neuron.NeuronLayer(ml_nn, **opt_for_ml)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.pooling_layer)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)

