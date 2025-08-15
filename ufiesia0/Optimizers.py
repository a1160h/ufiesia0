# Optimizer
# 最適化関数
# 2025.08.15 A.Inoue

from ufiesia0.Config import *

#### 最適化関数の共通機能 ############################################
class OptimizerBase:
    ''' g_clipなどの共通機能を各optimizerに付与する '''
    def __init__(self, **kwargs): 
        self.gradient_clipping = GradientClipping()

    def update(self, gradient, **kwargs):
        eta    = kwargs.pop('eta',    0.01)
        g_clip = kwargs.pop('g_clip', None)
        eta = self.gradient_clipping(gradient, eta, g_clip)
        self.eta = eta
        y = self.__call__(gradient, eta)
        return y

class GradientClipping:
    def __call__(self, gradient, eta, g_clip):
        if g_clip is None: 
            return eta
        g_l2n = np.sqrt(np.sum(gradient ** 2)) # 勾配のL2ノルム
        rate = g_clip / (g_l2n + 1e-6)
        eta_hat = eta * rate if rate < 1 else eta
        return  eta_hat

#### 各種最適化関数 ##################################################
class SGD(OptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, gradient, eta):
        return eta * gradient

class Momentum(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum = kwargs.pop('momentum', 0.9)
        super().__init__(**kwargs)
        self.vlcty = None
        
    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        self.vlcty -= (1 - self.momentum) * (self.vlcty - gradient) # 移動平均　
        return eta * self.vlcty
        
class RMSProp(OptimizerBase):
    def __init__(self, **kwargs): 
        self.decayrate = kwargs.pop('decayrate', 0.9)
        super().__init__(**kwargs)
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.hstry is None:
            self.hstry = np.ones_like(gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - gradient ** 2) # 移動平均
        return eta * gradient / (np.sqrt(self.hstry) + 1e-7)
    
class AdaGrad(OptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient)
        self.hstry += gradient ** 2
        return eta * gradient / (np.sqrt(self.hstry) + 1e-7)

class Adam(OptimizerBase):
    def __init__(self, **kwargs): 
        self.momentum  = kwargs.pop('momentum',  0.9)
        self.decayrate = kwargs.pop('decayrate', 0.9)
        super().__init__(**kwargs)
        self.vlcty = None
        self.hstry = None

    def __call__(self, gradient, eta):
        if self.vlcty is None:
            self.vlcty = np.zeros_like(gradient)
        if self.hstry is None:
            self.hstry = np.zeros_like(gradient)
         
        self.vlcty -= (1 - self.momentum)  * (self.vlcty - gradient)
        self.hstry -= (1 - self.decayrate) * (self.hstry - gradient ** 2)

        return eta * self.vlcty / (np.sqrt(self.hstry) + 1e-7)

