# 基底関数の線形結合
# Linear combination of basis functions
# 2024.06.27 A.Inoue

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -- 基底関数：直線(1次式)基底 Rectilinear Basis [1, x]　
class RectilinearBasis:
    def __init__(self):
        print('Initialize RectilinearBasis')

    def __call__(self, x):
        #print('Call RectilinearBasis')
        y = np.array([x**0, x])                        # x**0=1だが、xがベクトルのため       
        y = y.T                                        # 軸0:データx、軸1:基底次元
        return y

# -- 基底関数：多項式基底 Polynorminal Basis [x**0, x**1, ･･･ ,x**(m-1)]　
class PolynorminalBasis:
    def __init__(self, m=2):
        self.m = m
        print('Initialize PolynorminalBasis')

    def __call__(self, x):
        #print('Call PolynorminalBasis')
        m = self.m
        #print(x, m)
        #x = np.array(x)
        y = np.array([x**j for j in range(m)])         # xの冪乗のベクトル
        y = y.T                                        # 軸0:データx、軸1:基底次元
        return y

# -- 基底関数：ガウス基底 Gaussian Basis
class GaussianBasis:
    def __init__(self, m=2, s=1.0, *mu):
        self.m = m
        self.s = s
        self.mu = mu
        print('Initialize GaussianBasis')
        if len(self.mu)>0 and len(self.mu)!=m:
            raise Exception('Inconsistent specification of "m" and "mu"') 

    def __call__(self, x):
        #print('Call GaussianBasis')
        m = self.m
        s = self.s
        if len(self.mu)==0:
            mu = np.linspace(np.min(x), np.max(x), m+2, endpoint=True)
            mu = mu[1:-1]                              # 次数+2で一旦生成して両端を除く
            self.mu = mu
            print('Initialize mu')
        else:
            mu = self.mu
            #print('Use mu already set')
        y_shape = len(x), m                            # 軸0:データx、軸1:基底次元  
        mu = np.broadcast_to(mu, y_shape)              # データx方向に拡張
        x  = np.broadcast_to(x.reshape(-1,1), y_shape) # 基底mu方向に拡張 
        y = np.exp(- (x - mu)**2 / (2 * s**2))         # ガウス分布 mu:中心、s:広がり
        return y

# -- 基底関数：sigmoid基底 Sigmoid Basis
class SigmoidBasis:
    def __init__(self, m=2, s=1.0, *mu):
        self.m = m
        self.s = s
        self.mu = mu
        print('Initialize SigmoidBasis')
        if len(self.mu)>0 and len(self.mu)!=m:
            raise Exception('Inconsistent specification of "m" and "mu"') 

    def __call__(self, x):
        #print('Call SigmoidBasis')
        m = self.m
        s = self.s
        if len(self.mu)==0:
            mu = np.linspace(np.min(x), np.max(x), m+2, endpoint=True)
            mu = mu[1:-1]                              # 次数+2で一旦生成して両端を除く
            self.mu = mu
            print('Initialize mu')
        else:
            mu = self.mu
            #print('Use mu already set')
        y_shape = len(x), m                            # 軸0:データx、軸1:基底次元  
        mu = np.broadcast_to(mu, y_shape)              # データx方向に拡張
        x  = np.broadcast_to(x.reshape(-1,1), y_shape) # 基底mu方向に拡張
        a = (x - mu) / s                               # mu:中心、s:広がり
        y = 1 / (1 + np.exp(-a))                       # sigmoid関数
        return y

class SineBasis:
    def __init__(self, m=2):
        self.m = m
        self.func = lambda k, x: np.sin(k*x)
        print('Initialize SineBasis')

    def __call__(self, x):
        m = self.m
        y = np.array([self.func(k, x) for k in range(1, m, 1)])
        y = y.T                          # 軸0:データx、軸1:基底次元
        ones = np.ones((len(x),1))       # 定数項
        y = np.hstack((ones, y))         # 左端は定数項
        return y

class CosineBasis:
    def __init__(self, m=2):
        self.m = m
        self.func = lambda k, x: np.cos(k*x)
        print('Initialize CosineBasis')

    def __call__(self, x):
        m = self.m
        y = np.array([self.func(k, x) for k in range(1, m, 1)])
        y = y.T                          # 軸0:データx、軸1:基底次元
        ones = np.ones((len(x),1))       # 定数項
        y = np.hstack((ones, y))         # 左端は定数項
        return y

class FourierBasis:
    def __init__(self, m=2, T=1):
        self.m = m
        self.func = lambda n, x: [np.cos(n*x/T), np.sin(n*x/T)] 
        print('Initialize FourierBasis')

    def __call__(self, x):
        m = self.m
        y = np.array([self.func(n, x) for n in range(1, m//2+1, 1)])
        y = y.reshape(-1, len(x)).T      # 次元を１つ減らして転置
        ones = np.ones((len(x), 1))      # 定数項
        y = np.hstack((ones, y))         # 左端は定数項
        y = y[:, :m]
        return y

class LinearCombination: # Linear Combination of Basis Functuions
    def __init__(self, basis_function=None):
        self.phi = basis_function
        self.w = None
        print('Initialize Linear Combination of', self.phi.__class__.__name__)    
    
    def __call__(self, x):
        if self.w is None:
            raise Exception('Need to fix parametwers')
        self.Pi = self.phi(x)
        y = np.dot(self.Pi, self.w)
        return y

    def backward(self, gy=1):
        #gw = np.dot(self.Pi.T, gy) / len(gy)
        gw = np.dot(gy, self.Pi) 
        return gw

    def regression(self, x, t, r=0.0):
        if self.phi is None:
            raise Exception('Need to initialize specifing basis function')
        Pi = self.phi(x)#.astype('float')
        tPiPi = np.dot(Pi.T, Pi)
        I = np.eye(len(tPiPi))  
        tPiPi += r * I        # 正則化項を加える　　
        tPiPiI = np.linalg.inv(tPiPi)
        PiI = np.dot(tPiPiI, Pi.T)
        w = np.dot(PiI, t)
        print('Parameters are fixed.')
        self.w = w
        return w

    def get(self):
        def func(x):
            Pi = self.phi(x)
            y = np.dot(Pi, self.w)
            return y
        return func

#class LCBF: # Linear Combination of Basis Functuions
class LinearCombination2: # Linear Combination of Basis Functuions
# 線形回帰の関数：phi0(x)*W[0] + phi1(x)*W[1] + ・・・ + phi(m-1)(x)*W[m-1]
# Pi = Σphi(x)    
    def __init__(self, BF=None, dim=2, s=1.0, *mu):
        if BF in('Linear', 'linear'):
            self.phi = RectilinearBasis()
        elif BF in ('Gauss', 'gauss', 'Gaussian', 'gaussian'):
            self.phi = GaussianBasis(dim, s, *mu)
        elif BF in ('Sigmoid', 'sigmoid'):
            self.phi = SigmoidBasis(dim, s, *mu)
        else:
            self.phi = PolynorminalBasis(dim)
        print('Initialize Linear Combination of Basis Functions', self.phi)    
    
    def forward(self, x, w):
        self.Pi = self.phi(x)
        y = np.dot(self.Pi, w)
        return y

    def backward(self, gy=1):
        #gw = np.dot(self.Pi.T, gy) / len(gy)
        gw = np.dot(gy, self.Pi) 
        return gw

# -- 二乗和誤差 --
class MeanSquaredError:
    def forward(self, y, t):
        self.inputs = y, t
        return 0.5 * np.sum((y - t) ** 2)/len(y)

    def backward(self, gl=1):
        y, t = self.inputs
        gl /= len(y)
        #gl = np.broadcast_to(np.array(gl), y.shape)
        gy = gl * (y - t) 
        return gy 

# -- 二乗和誤差 --
class MeanSquaredErrorDecay:
    def forward(self, y, t, r, w):
        self.inputs = y, t, r, w
        loss= 0.5 * np.sum((y - t) ** 2)
        decay = r * 0.5 * np.sum(np.square(w))
        return (loss + decay) / len(y)

    def backward(self, gl=1):
        y, t, r, w = self.inputs
        gl /= len(y)
        #gl = np.broadcast_to(np.array(gl), y.shape)
        gy = gl * (y - t)
        gw = r * gl * w 
        return gy, gw

## -- 数値微分 --
def numerical_gradient(func, x, h=1e-7):  # 数値微分、変数xは配列に対応　
    grad = np.empty(x.shape)
    for i, xi in enumerate(x):            # xの要素を順に選んで±h
        x[i] = xi + h                     # 要素x[i]を+h、他はそのまま 
        fxph = func(x)
        x[i] = xi - h                     # 要素x[i]を-h、他はそのまま　
        fxmh = func(x)                       
        grad[i] = (fxph - fxmh) / (2 * h)  
        x[i] = xi                         # x[i]をもとに戻す 
    return grad

# -- 線形回帰の1次関数(係数をwとする) --
class LinearFunction:
    def __init__(self):
        print('Initialize LinearFunction', __class__)
        
    def forward(self, x, w):
        self.w = w
        y = 0
        X = [x ** i for i in range(len(w))] # 基底関数としてxの冪乗の並びを作る
        X = np.array(X)
        X = X.T                             # データ数を0軸、冪乗数を1軸にする
        self.X = X
        y = np.dot(X, w)                    # <- この逆伝播ならば簡単
        return y                       

    def backward(self, gl=1):
        X = self.X
        gw = np.dot(X.T, gl) / len(X)
        return gw

def sum_squared_error(y, t):
    return 1/2 * np.sum(np.square(y - t))

def graph3d(func, rx=(0,1), ry=(0,1), label=None):
    qx = np.linspace(*rx, 100)
    qy = np.linspace(*ry, 100)
    X, Y, Z = [], [], []
    for x in qx:
        for y in qy:
            z = func(x, y)
            X.append(x)
            Y.append(y)
            Z.append(z)
            
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    if label is not None:
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])
    ax.scatter(X, Y, Z, c='r', s=2, marker='.')
    plt.show()
   
def graph3dc(func, rx=(0,1), ry=(0,1), label=None, cm='gnuplot'):
    qx = np.linspace(*rx, 100)
    qy = np.linspace(*ry, 100)
    X, Y, Z = [], [], []
    for x in qx:
        for y in qy:
            z = func(x, y)
            X.append(x)
            Y.append(y)
            Z.append(z)
            
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    if label is not None:
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])
    cm = plt.cm.get_cmap(cm)
    ax.scatter(X, Y, Z, c=Z, cmap=cm, marker='.')
    plt.show()

    # おまけ
    ZA = np.array(Z)
    argminZA = np.argmin(ZA)
    print(argminZA, Z[argminZA], X[argminZA], Y[argminZA])
    
def graph3d4(func, rx0=(0,0), rx1=(-10,10), rx2=(-10,10), label=None):
    qx0 = np.linspace(*rx0, 30)
    qx1 = np.linspace(*rx1, 30)
    qx2 = np.linspace(*rx2, 30)
    X0, X1, X2, Z = [], [], [], []

    for x0 in qx0:
        for x1 in qx1:
            for x2 in qx2:
                z = func(x0, x1, x2)
                X0.append(x0)
                X1.append(x1)
                X2.append(x2)
                Z.append(z)

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(X1, X2, Z, marker='.', s=2, c='red')
    plt.show()
