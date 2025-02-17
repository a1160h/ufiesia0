# 20210918 A.Inoue
# 行列の操作の関数を定義します

import copy
import math

# データを入力　数値ひとつだけならベクトル、数値２つなら行列
def get(*size):
    '''大きさを指定して、その大きさの行列やベクトルにデータを入力する'''
    if type(size[0]) in (tuple,list):
        size, = size
    if len(size) > 1:    
        X = [[0 for j in range(size[1])] for i in range(size[0])]
        for i in range(size[0]):
            for j in range(size[1]):
                X[i][j] = float(input(str(i+1)+'行'+str(j+1)+'列のデータ'))
    else:
        X = [0.0 for i in range(size[0])]
        for i in range(size[0]):
            X[i] = float(input(str(i+1)+'番目のデータ'))
    return X

# 大きさ
def size(X):
    '''ベクトルや行列の大きさを読み取る（0はその次元がない）'''
    if type(X) == list and type(X[0]) == list:
        mx = len(X)
        nx = len(X[0])
        #print('X is a matrix')
    elif type(X) == list:
        mx = 0
        nx = len(X)
        #print('X is a vector')
    else:
        mx = 0
        nx = 0
        #print('X is a scalar')
    return mx, nx

# 三角関数
def sin(X):
    return op1(lambda x: math.sin(x), X)

def cos(X):
    return op1(lambda x: math.cos(x), X)

def tan(X):
    return op1(lambda x: math.tan(x), X)

def asin(X):
    return op1(lambda x: math.asin(x), X)

def acos(X):
    return op1(lambda x: math.acos(x), X)

def atan(X):
    return op1(lambda x: math.atan(x), X)

def sinh(X):
    return op1(lambda x: math.sinh(x), X)

def cosh(X):
    return op1(lambda x: math.cosh(x), X)

def tanh(X):
    return op1(lambda x: math.tanh(x), X)

def asinh(X):
    return op1(lambda x: math.asinh(x), X)

def acosh(X):
    return op1(lambda x: math.acosh(x), X)

def atanh(X):
    return op1(lambda x: math.atanh(x), X)

# その他の単一オペランド演算
def sqrt(X):
    return op1(lambda x: math.sqrt(x), X)


# 四則演算
def add(X,Y):
    return ops(lambda x, y: x + y, X, Y)

def sub(X,Y):
    return ops(lambda x, y: x - y, X, Y)

def mul(X,Y):
    return ops(lambda x, y: x * y, X, Y)

def div(X,Y):
    return ops(lambda x, y: x / y, X, Y)

# 指数、対数
def exp(X,Y=math.e):#2.718281828459045235360287471352):
    return ops(lambda x, y: y**x,  X, Y)

def log(X,Y=math.e):#2.718281828459045235360287471352):
    return ops(lambda x, y: math.log(x, y), X, Y)

# 比較
def eq(X,Y):
    return ops(lambda x, y: x == y, X, Y)

def neq(X,Y):
    return ops(lambda x, y: x != y, X, Y)

def gt(X,Y):
    return ops(lambda x, y: x > y, X, Y)

def gte(X,Y):
    return ops(lambda x, y: x >= y, X, Y)

def lt(X,Y):
    return ops(lambda x, y: x < y, X, Y)

def lte(X,Y):
    return ops(lambda x, y: x <= y, X, Y)

# 要素の演算(単一オペランド)
def op1(func, X):
    '''ベクトルや行列の要素の演算(オペランドは１つ)'''
    mx, nx = size(X)

    if   mx == 0 and nx == 0: # スカラ    
        Y = func(X)

    elif mx == 0 and nx > 0:  # ベクトル
        Y = [func(x) for x in X]

    else:                     # 行列(縦ベクトルを含む) 
        Y = [[func(x) for x in vx] for vx in X]

    return Y    

# 要素同士の演算
def ops(func, X, Y):
    '''ベクトルや行列の要素どうしの演算(大きさが違う場合には拡張して実行)'''
    mx, nx = size(X)
    my, ny = size(Y)
    
    if   mx == my and nx == ny and mx > 0:  # 行列＠行列 
        Z = [[func(x, y) for x, y in zip(vx, vy)] for vx, vy in zip(X, Y)]
        
    elif mx == my and nx == ny and nx > 0:  # ベクトル＠ベクトル
        Z = [func(x, y) for x, y in zip(X, Y)]
        
    elif mx == my and nx == ny:             # スカラ＠スカラ
        Z = func(X, Y)

    elif mx == my and mx > 0 and ny == 1:   # 行列＠縦ベクトル拡張 
        Z = [[func(x, vy[0]) for x in vx] for vx, vy in zip(X, Y)]

    elif mx == my and mx > 0 and nx == 1:   # 縦ベクトル拡張＠行列 
        Z = [[func(vx[0], y) for y in vy] for vx, vy in zip(X, Y)]
        
    elif mx > 0 and my == 0 and nx == ny:   # 行列＠ベクトル拡張
        Z = [[func(x, y) for x, y in zip(vx, Y)] for vx in X]

    elif mx == 0 and my > 0 and nx == ny:   # ベクトル拡張＠行列 
        Z = [[func(x, y) for x, y in zip(X, vy)] for vy in Y]

    elif mx > 0 and my == 1 and ny == 1:    # 行列＠１要素の行列　
        Z = [[func(x, Y[0][0]) for x in vx] for vx in X]
                
    elif mx > 0 and my == 0 and ny == 1:    # 行列＠１要素のベクトル　
        Z = [[func(x, Y[0]) for x in vx] for vx in X]

    elif mx > 0 and my == 0 and ny == 0:    # 行列＠スカラ
        Z = [[func(x, Y) for x in vx] for vx in X]

    elif my > 0 and mx == 1 and nx == 1:    # １要素の行列＠行列　
        Z = [[func(X[0][0], y) for y in vy] for vy in Y]
                
    elif my > 0 and mx == 0 and nx == 1:    # １要素のベクトル＠行列　
        Z = [[func(X[0], y) for y in vy] for vy in Y]

    elif my > 0 and mx == 0 and nx == 0:    # スカラ＠行列
        Z = [[func(X, y) for y in vy] for vy in Y]

    elif mx == 0 and nx > 0 and my == 0 and ny == 1: # ベクトル＠１要素のベクトル　
        Z = [func(x, Y[0]) for x in X]

    elif mx == 0 and nx > 0 and my == 0 and ny == 0: # ベクトル＠スカラ
        Z = [func(x, Y) for x in X]

    elif my == 0 and ny > 0 and mx == 0 and nx == 1: # １要素のベクトル＠ベクトル　
        Z = [func(X[0], y) for y in Y]

    elif my == 0 and ny > 0 and mx == 0 and nx == 0: # スカラ＠ベクトル
        Z = [func(X, y) for y in Y]

    else:
        raise Exception('Shape of operand is wrong')

    return Z    
  
# 内積、行列積
def dot(X,Y):
    '''行列積やベクトルの内積(大きさが違う場合には拡張して実行)'''
    mx, nx = size(X)
    my, ny = size(Y)
    
    if mx > 0 and my > 0 and nx == my:     # 行列・行列 
        XY = [[0.0 for j in range(ny)] for i in range(mx)]
        for i in range(mx): 
            for j in range(ny):
                for k in range(my):
                    XY[i][j] += X[i][k] * Y[k][j]

    elif mx == 0 and my == 0 and nx == ny: # ベクトル内積 
        XY = 0.0 
        for i in range(nx):
            XY += X[i] * Y[i]

    elif mx > 0 and my == 0 and nx == ny:  # 行列・ベクトル(縦ベクトルとして扱う) 
        XY = [ 0.0 for i in range(mx)] 
        for i in range(mx):
            for k in range(ny):
                XY[i] += X[i][k] * Y[k]

    elif mx > 0 and nx == 1 and my == 0:   # 縦ベクトル・横ベクトル(横ベクトルを転置しない) 
        XY = [[0.0 for k in range(ny)] for i in range(mx)]
        for i in range(mx):
            for k in range(ny):
                XY[i][k] += X[i][0] * Y[k]

    elif mx == 0 and my > 0 and nx == my:  # ベクトル・行列(ベクトルは横ベクトルのまま)
        XY = [ 0.0 for i in range(ny)] 
        for i in range(ny):
            for j in range(nx):
                XY[i] += X[j] * Y[j][i]

    else:
        raise Exception('Shape of operand is wrong')
        
    return XY    

# 行列の転置 [u行v列]→[v行u列]
def trn(X):
    '''行列の転置'''
    m, n = size(X)
    
    if m > 0: # 行列
        Z = [[X[j][i] for j in range(m)] for i in range(n)]
    else:     # ベクトル
        Z = [[X[i]] for i in range(n)]
        
    return Z 
            
# 単位行列
def eye(n):
    '''単位行列の生成'''
    return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]


# 逆行列
def inv(X):
    '''逆行列を求める'''
    m, n = size(X)
    Y = eye(n)
    Z = copy.deepcopy(X)

    if m != n:
        raise Exception('Shape of operand is wrong', Z)

    for k in range(n):
        #print('Check singular matrix')
        if Z[k][k]==0.0:
            raise Exception('Singular matrix', Z)
        
    # 下半分
    for k in range(n-1):             # kで注目する対角要素を指定
        for i in range(k+1,n):       # iで操作する行を指定
            r = Z[i][k] / Z[k][k]    # i行からk行のz倍を引く
            for j in range(n):       # jを回して行内の要素を加工
                Z[i][j] = Z[i][j] - Z[k][j] * r
                Y[i][j] = Y[i][j] - Y[k][j] * r

    # 上半分
    for k in reversed(range(1,n)):   # kで注目する対角要素を指定
        for i in reversed(range(k)): # iで操作する行を指定  
            r = Z[i][k] / Z[k][k]    # i行からk行のz倍を引く
            for j in range(n):       # jを回して行内の要素を加工
                Z[i][j] = Z[i][j] - Z[k][j] * r
                Y[i][j] = Y[i][j] - Y[k][j] * r

    # 対角要素を１に
    for i in range(n):
        r = 1 / Z[i][i]
        for j in range(n):
            Z[i][j] = Z[i][j] * r
            Y[i][j] = Y[i][j] * r

    return Y  # 逆行列        

# 勾配法による逆行列
def dinv(X, lr=0.06, itr=5000, g_clip=100):
    '''勾配法による逆行列'''
    m, n = size(X)
    E = eye(m)
    Y = [[1.0 for j in range(m)] for i in range(n)]
    for i in range(itr):
        XY = dot(X, Y)
        XYmnE = sub(XY, E)
        #XYmnE2 = exp(2, XYmnE)    # (XY - E)**2 
        #L = 0.5 * Sum(XYmnE2)     # 二乗和誤差
        l2n = Sum(exp(2, XYmnE)) ** 0.5 # L2 norm
        rate = g_clip / (l2n + 1e-7) if l2n >= g_clip else 1
        dLdB = dot(trn(X), XYmnE)
        Y = sub(Y, mul(lr*rate, dLdB))
    return Y

# ムーア・ペンローズの一般逆行列
def pinv(X):
    '''ムーア・ペンローズの一般逆行列'''
    tX = trn(X)
    return dot(inv(dot(tX, X)), tX) 

# linspace
def linspace(start, stop, num=50):
    '''startからstopまでを50等分にして並べる'''
    step = (stop - start) / (num - 1)
    y = []
    for i in range(num):
        y.append(start + step * i)
    return y

# sum
def Sum(X):
    '''ベクトルや行列の要素の全合計'''
    m, n = size(X)
    s = 0
    if m > 0:
        for i in range(m):
            s += sum(X[i])
    else:
        s = sum(X)
    return s    
    
# max
def max(X):
    m, n = size(X)
    if m == 0: # ベクトル
        max_x = X[0]
        for i in range(n):
            x = X[i]
            if x > max_x:
                max_x = x
    if m > 0: # 行列
        max_x = X[0][0]
        for i in range(m):
            for j in range(n):
                x = X[i][j]
                if X[i][j] > max_x:
                    max_x = x
    return max_x                                   

# min
def min(X):
    m, n = size(X)
    if m == 0: # ベクトル
        min_x = X[0]
        for i in range(n):
            x = X[i]
            if x < min_x:
                min_x = x
    if m > 0: # 行列
        min_x = X[0][0]
        for i in range(m):
            for j in range(n):
                x = X[i][j]
                if X[i][j] < min_x:
                    min_x = x
    return min_x                                   
    
# argmax
def argmax(X):
    m, n = size(X)
    if m == 0: # ベクトル
        max_x = X[0]; max_i = 0
        for i in range(n):
            x = X[i]
            if x > max_x:
                max_i = i
                max_x = x
        argmax = max_i        
    if m > 0: # 行列
        max_x = X[0][0]; max_i = 0, 0
        for i in range(m):
            for j in range(n):
                x = X[i][j]
                if X[i][j] > max_x:
                    max_i = i, j
                    max_x = x
        argmax = max_i[0] * n + max_i[1]             
    return argmax                                   
        
# argmin
def argmin(X):
    m, n = size(X)
    if m == 0: # ベクトル
        min_x = X[0]; min_i = 0
        for i in range(n):
            x = X[i]
            if x < min_x:
                min_i = i
                min_x = x
        argmin = min_i        
    if m > 0: # 行列
        min_x = X[0][0]; min_i = 0, 0
        for i in range(m):
            for j in range(n):
                x = X[i][j]
                if X[i][j] < min_x:
                    min_i = i, j
                    min_x = x
        argmin = min_i[0] * n + min_i[1]             
    return argmin                                   
        
# where
def where(X, Y, Z):
    '''
    X * Y + (1 - X) * Z
    X : condition
    Y : true
    Z : false
    '''
    XY = mul(X, Y)
    negX = sub(1, X)
    negXZ = mul(negX, Z)
    return add(XY, negXZ)


def cross_concatenate(X, Y):
    '''
    XとYを余白を0で埋めて斜めにクロスして連結

    X 0
    0 Y 

    '''
    if len(X) == 0:
        return Y
    if len(Y) == 0:
        return X

    mx, nx = size(X)
    my, ny = size(Y)
    Z = [] 
    for i in range(mx):
        Z.append(X[i] + [0.0 for j in range(ny)])
    for i in range(my):
        Z.append([0.0 for j in range(nx)] + Y[i])
    return Z    

def v_stack(X, Y):
    '''
    XとYを余白を0で埋めて列数を合わせて縦に連結

    X<Y    X>Y　　

　　-->    　
    X 0    X
    Y      Y 0
           -->
    '''
    if len(X) == 0:
        return Y
    if len(Y) == 0:
        return X

    mx, nx = size(X)
    my, ny = size(Y)
    Z = [] 
    if nx < ny:
        for i in range(mx):
            Z.append(X[i] + [0.0 for j in range(ny - nx)])
        Z = Z + Y
    if nx >= ny:
        for i in range(my):
            Z.append(Y[i] + [0.0 for j in range(nx - ny)])
        Z = X + Z
    return Z    

def h_stack(X, Y):
    '''
    XとYを余白を0で埋めて行数を合わせて横に連結

　　 X<Y    X>Y
　
    |X Y    X Y|
    V0        0V
           
    '''
    if len(X) == 0:
        return Y
    if len(Y) == 0:
        return X

    mx, nx = size(X)
    my, ny = size(Y)
    Z = [] 
    if mx < my:
        for i in range(mx):
            Z.append(X[i] + Y[i])
        for i in range(mx, my):
            Z.append([0.0 for j in range(nx)] + Y[i])
    if mx >= my:
        for i in range(my):
            Z.append(X[i] + Y[i])
        for i in range(my, mx):
            Z.append(X[i] + [0.0 for j in range(ny)])
    return Z    

def sigmoid(X, base=math.e):# base=2.718281828459045235360287471352):
    Y = mul(X, -1)
    Y = exp(Y, base)
    Y = add(1, Y)
    Y = div(1, Y)
    return Y

def tanh(X, base=math.e):# base=2.718281828459045235360287471352):
    nX = mul(X, -1)
    eX = exp(X, base) 
    enX = exp(nX, base)
    Y = div(sub(eX, enX), add(eX, enX))
    return Y

def swish(X, base=math.e):# base=2.718281828459045235360287471352):
    Y = mul(X, -1)
    Y = exp(Y, base)
    Y = add(1, Y)
    Y = div(X, Y)
    return Y

def mat2vec(X):
    '''
    convert square matrix to vector
    
    '''
    mx, ny = size(X)
    Y = [X[i][i] for i in range(mx)]
    return Y    

def vec2mat(X):
    '''
    convert vector to square matrix
    
    '''
    Y = [[0.0 for x in X] for x in X]
    for i, x in enumerate(X):
        Y[i][i] = x
    return Y

    
