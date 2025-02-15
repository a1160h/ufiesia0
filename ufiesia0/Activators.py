# 2022.05.20 井上
from ufiesia0.Config import *
print(np.__name__, 'is running in', __file__, np.random.rand(1))    

#### 活性化関数 ######################################################
class Identity:
    def forward(self, x):
        return x 
 
    def backward(self, gy): 
        return gy

class Step:
    def __init__(self, t=0):
        self.t = t
        
    def forward(self, x):
        y = x > self.t
        return y 

class Sigmoid:
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, gy):
        y = self.y         
        gx = y * (1 - y) * gy
        return gx

class Softmax:
    def forward(self, x):
        max_x = np.max(x, axis=-1, keepdims=True) 
        exp_a = np.exp(x - max_x)  # オーバーフロー対策
        sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)  
        self.y = exp_a / (sum_exp_a + 1e-7)
        return self.y

    def backward(self, gy): 
        y = self.y
        gx = y * gy
        sumgx = np.sum(gx, axis=-1, keepdims=True)
        gx -= y * sumgx 
        return gx

class Tanh:
    def forward(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def backward(self, gy):
        y = self.y
        gx = (1 - y * y) * gy
        return gx

class ReLU:
    def forward(self, x):
        y = np.where(x<=0, 0, x)
        self.x = x
        return y

    def backward(self, gy):
        x  = self.x
        gx = gy * np.where(x<=0, 0, 1) 
        return gx

class LReLU():
    def __init__(self, **kwargs):
        self.c = kwargs.pop('c', 0.01)
        
    def forward(self, x):
        y = np.where(x>=0, x, self.c * x)
        self.x = x
        return y

    def backward(self, gy):
        x  = self.x
        gx = gy * np.where(x>=0, 1, self.c)
        return gx
