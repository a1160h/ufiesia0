# LossFunctions
# 2022.05.20 井上
from ufiesia0.Config import *
print(np.__name__, 'is running in', __file__, np.random.rand(1))    

class MeanSquaredError:
    def forward(self, y, t):
        self.y = y
        self.t = t
        self.k = y.size // y.shape[-1]
        l = 0.5 * np.sum((y - t) ** 2)
        return l / self.k 

    def backward(self, gl=1):
        y = self.y
        t = self.t
        gy = gl * (y - t)
        return gy / self.k
   
class CrossEntropyError:
    def forward(self, y, t):
        self.y = y
        self.t = t
        self.k = y.size // y.shape[-1]  # 時系列データで必須
        l = - np.sum(t * np.log(y + 1e-7))
        return l / self.k

    def backward(self, gl=1):
        y = self.y
        t = self.t
        gy = - gl * t / (y + 1e-7)
        return gy / self.k

class CrossEntropyError2():
    def forward(self, y, t):
        self.y = y
        self.t = t
        self.k = y.size // y.shape[-1]  # 時系列データで必須
        l = -np.sum(t*np.log(y+1e-7)+(1-t)*np.log(1-y+1e-7))
        return l / self.k

    def backward(self, gl=1):
        y = self.y
        t = self.t
        gy = gl * (y - t) / (y * (1 - y) + 1e-7)
        return gy / self.k

