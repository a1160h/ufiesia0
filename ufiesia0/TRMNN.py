#import pickle
from ufiesia0.Config import *
from ufiesia0 import Neuron, LossFunctions, NN
from ufiesia0 import common_function as cf


class TRMNN(NN.NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml_nn = kwargs.pop('ml_nn', 100)                    
        opt_for_ml = {}
        opt_for_ol = {}
        opt_for_ml['activate'] = kwargs.pop('ml_act', 'Identity')
        opt_for_ol['activate'] = kwargs.pop('ol_act', 'Identity')
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.markov_layer  = Neuron.SimpleRnnLayer(In, ml_nn, **opt_for_ml)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.markov_layer)
        self.layers.append(self.output_layer)

    def forward(self, x, r, t=None):
        y = self.markov_layer.forward(x, r)
        y = self.output_layer.forward(y)
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
        gx = self.output_layer.backward(gy)
        gx, gr = self.markov_layer.backward(gx)
        return gx, gr
    
class TRMNN2(NN.NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml_nn = kwargs.pop('ml_nn', 100)                    
        opt_for_ml = {}
        opt_for_ol = {}
        opt_for_ml['activate'] = kwargs.pop('ml_act', 'Identity')
        opt_for_ol['activate'] = kwargs.pop('ol_act', 'Identity')
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.markov_layer  = Neuron.SimpleRnnLayer(In, ml_nn, **opt_for_ml)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.markov_layer)
        self.layers.append(self.output_layer)

        self.reset_state()

    def forward(self, x, t=None):
        r0 = np.zeros_like(x)
        r = r0 if self.r is None else self.r
        #print('>>>', x.shape, r.shape)
        y = self.markov_layer.forward(x, r)
        y = self.output_layer.forward(y)
        self.r = x
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
        gx = self.output_layer.backward(gy)
        gx, gr = self.markov_layer.backward(gx)
        return gx, gr

    def reset_state(self):
        self.r = None
        
class TRMNN3(NN.NN_CNN_Base):
    def __init__(self, *args, **kwargs):
        In, Out = super().__init__(*args, **kwargs)
        ml_nn = kwargs.pop('ml_nn', 100)                    
        opt_for_ml = {}
        opt_for_ol = {}
        opt_for_ml['activate'] = kwargs.pop('ml_act', 'Identity')
        opt_for_ol['activate'] = kwargs.pop('ol_act', 'Identity')
        opt_for_ml.update(kwargs)
        opt_for_ol.update(kwargs) 

        # -- 各層の初期化 -- 
        self.markov_layer  = Neuron.SimpleRnnLayer(In, ml_nn, **opt_for_ml)
        self.output_layer  = Neuron.NeuronLayer(Out, **opt_for_ol)

        # -- layerのまとめ -- 
        self.layers.append(self.markov_layer)
        self.layers.append(self.output_layer)

        self.reset_state()

    def forward(self, x, t=None):
        r0 = np.zeros_like(x)
        r = r0 if self.r is None else self.r
        #print('>>>', x.shape, r.shape)
        y = self.markov_layer.forward(x, r)
        y = self.output_layer.forward(y)
        self.r = y
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
        gx = self.output_layer.backward(gy)
        gx, gr = self.markov_layer.backward(gx)
        return gx, gr

    def reset_state(self):
        self.r = None
