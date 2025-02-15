import pickle
from ufiesia0.Config import *
from ufiesia0 import Neuron, LossFunctions
from ufiesia0 import common_function as cf

class NN_CNN_Base:
    def __init__(self, *args, **kwargs):
        print(args, kwargs)
        # 入出力の判定
        if len(args) == 2:
            In, Out = args
        elif len(args) == 0:
            In = None; Out = None
        else:
            In = None; Out = args[-1]
        print(args, 'number of output =', Out)
        self.layers = []

        # 損失関数　
        loss_function_name = kwargs.pop('loss', 'MeanSquaredError')
        self.loss_function = cf.eval_in_module(loss_function_name, LossFunctions)

        return In, Out
        
    def summary(self):
        print('～～ model summary of', self.__class__.__name__, '～'*24)
        for i, layer in enumerate(self.layers):
            print('layer', i, layer.__class__.__name__)
            print(' configuration =', layer.config, end='')
            if hasattr(layer, 'activator'):
                print('\n activate =', layer.activator.__class__.__name__, end=' ')
            print('\n' + '-'*72)
        if hasattr(self, 'loss_function'):
            print('loss_function =', self.loss_function.__class__.__name__)
        print('～～ end of summary ' + '～'*28 + '\n')

    def forward(self, x, t=None):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        if t is None:
            return y
        elif hasattr(self, 'loss_function'):
            l = self.loss_function.forward(y, t)
            return y, l
        else:
            raise Exception("Can't get loss by forward.")

    def backward(self, gy=None, gl=1):
        if gy is not None:
            pass
        elif self.loss_function.t is not None:
            gy = self.loss_function.backward(gl)
        else:
            raise Exception("Can't get gradient for backward." \
                            , 'gy =', gy, 'gl =', gl)
        gx = gy
        for layer in reversed(self.layers):
            gx = layer.backward(gx)
        return gx

    def update(self, **kwargs):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(**kwargs)

    def export_params(self):
        """ パラメタから辞書 """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                params['layer'+str(i)+'_w'] = np.array(layer.w)
            if hasattr(layer, 'v'):
                params['layer'+str(i)+'_v'] = np.array(layer.v) # RNN用
            if hasattr(layer, 'b'):
                params['layer'+str(i)+'_b'] = np.array(layer.b)
        return params

    def import_params(self, params):
        """ 辞書からパラメタ """
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                layer.w = np.array(params['layer'+str(i)+'_w']) 
            if hasattr(layer, 'v'):
                layer.v = np.array(params['layer'+str(i)+'_v']) # RNN用 
            if hasattr(layer, 'b'):
                layer.b = np.array(params['layer'+str(i)+'_b'])

    def save_parameters(self, file_name):
        """ 学習結果の保存 """
        title = self.__class__.__name__
        params = self.export_params()
        params['title'] = title 
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print(title, 'モデルのパラメータをファイルに記録しました=>', file_name)    

    def load_parameters(self, file_name):
        """ 学習結果の保存 """
        title = self.__class__.__name__
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        title_f = params.pop('title', None)
        print(title_f, 'モデルのパラメータをファイルから取得しました<=', file_name)
        if title_f == title:
            self.import_params(params)
            print('パラメータが継承されました')
        else:
            print('!!構成が一致しないためパラメータは継承されません!!')
        return params

class NN_0(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        self.layer = Neuron.NeuronLayer(Out, **kwargs)
        self.layers.append(self.layer)

class NN_1(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml_nn = kwargs.pop('ml_nn', 3)
        self.middle_layer = Neuron.NeuronLayer(ml_nn,  **kwargs)
        self.output_layer = Neuron.NeuronLayer(Out, **kwargs)
        self.layers.append(self.middle_layer)    
        self.layers.append(self.output_layer)

class NN_m(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml_nn = kwargs.pop('ml_nn', 3)                    
        opt_for_ml = {}
        opt_for_ol = {}
        opt_for_ml['activate'] = kwargs.pop('ml_act', 'Identity')
        opt_for_ol['activate'] = kwargs.pop('ol_act', 'Identity')
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.middle_layer  = Neuron.NeuronLayer(ml_nn, **opt_for_ml)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer)
        self.layers.append(self.output_layer)

class NN_mm(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml1_nn = kwargs.pop('ml1_nn', 3)                    
        ml2_nn = kwargs.pop('ml2_nn', 3)                    
        opt_for_ml1 = {}
        opt_for_ml2 = {}
        opt_for_ol = {}
        opt_for_ml1['activate'] = kwargs.pop('ml1_act', 'Identity')
        opt_for_ml2['activate'] = kwargs.pop('ml2_act', 'Identity')
        opt_for_ol['activate'] = kwargs.pop('ol_act', 'Identity')
        opt_for_ml1.update(kwargs)
        opt_for_ml2.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.middle_layer1 = Neuron.NeuronLayer(ml1_nn, **opt_for_ml1)
        self.middle_layer2 = Neuron.NeuronLayer(ml2_nn, **opt_for_ml2)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer1)
        self.layers.append(self.middle_layer2)
        self.layers.append(self.output_layer)

class NN_mmm(NN_CNN_Base): 
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml1_nn = kwargs.pop('ml1_nn', 3)                    
        ml2_nn = kwargs.pop('ml2_nn', 3)                    
        ml3_nn = kwargs.pop('ml3_nn', 3)                    
        opt_for_ml1 = {}
        opt_for_ml2 = {}
        opt_for_ml3 = {}
        opt_for_ol = {}
        opt_for_ml1['activate'] = kwargs.pop('ml1_act', 'Identity')
        opt_for_ml2['activate'] = kwargs.pop('ml2_act', 'Identity')
        opt_for_ml3['activate'] = kwargs.pop('ml3_act', 'Identity')
        opt_for_ol['activate'] = kwargs.pop('ol_act', 'Identity')
        opt_for_ml1.update(kwargs)
        opt_for_ml2.update(kwargs)
        opt_for_ml3.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.middle_layer1 = Neuron.NeuronLayer(ml1_nn, **opt_for_ml1)
        self.middle_layer2 = Neuron.NeuronLayer(ml2_nn, **opt_for_ml2)
        self.middle_layer3 = Neuron.NeuronLayer(ml3_nn, **opt_for_ml3)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.middle_layer1)
        self.layers.append(self.middle_layer2)
        self.layers.append(self.middle_layer3)
        self.layers.append(self.output_layer)

class NN_em(NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        emb_m = kwargs.pop('emb_m', In)                    
        emb_n = kwargs.pop('emb_n', 3)                    
        ml_nn = kwargs.pop('ml_nn', 3)                    
        opt_for_emb = {}
        opt_for_ml = {}
        opt_for_ol = {}
        opt_for_ml['activate'] = kwargs.pop('ml_act', 'Identity')
        opt_for_ol['activate'] = kwargs.pop('ol_act', 'Identity')
        opt_for_emb.update(kwargs)
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.embedding_layer = Neuron.Embedding(emb_m, emb_n, **opt_for_emb)
        self.middle_layer    = Neuron.NeuronLayer(ml_nn, **opt_for_ml)
        self.output_layer    = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.embedding_layer)
        self.layers.append(self.middle_layer)
        self.layers.append(self.output_layer)
    
