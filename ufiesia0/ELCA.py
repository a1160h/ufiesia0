# Electric Circuit Analysis
# 20210828 A.Inoue

from ufiesia import MatMath as mm
import copy
import pickle
import math

class ElectricCircuit:
        
    def set(self, diode=False):
        self.diode=diode
        t = int(input('Number of Transistors '))
        f = int(input('Number of FETs        '))
        n = int(input('Number of nodes       '))
        b = int(input('Number of branches    '))
        print('node# ', n, 'branch# ', b, 'transistor# ', t, 'FET# ', f)
        self.size = n, b, t, f

        Ab = set_A(n, b)
        At = set_A_trs(n, t)
        Af = set_A_fet(n+t, f) # <<<<要注意>>>>：FETはトランジスタの後ろにノード追加
        Yb, Eb, pnb = set_branch(b, diode)
        Yt, Et, pnt = set_branch_trs(t, diode)
        Yf, Ef, pnf = set_branch_fet(f, diode)
        
        A = mm.v_stack(Ab, At)
        A = mm.v_stack(A, Af)
        Y = mm.cross_concatenate(Yb, Yt)
        Y = mm.cross_concatenate(Y, Yf)
        E = Eb + Et + Ef
        pn = pnb + pnt + pnf
        
        self.A = A
        self.Y = Y
        self.E = E
        self.pn = pn

        return A, Y, E, pn
        
    def analize(self):
        A = self.A
        Y = self.Y
        E = self.E
        W, V, I, P = Analize(A, Y, E)
        self.W = W
        self.V = V
        self.I = I
        self.P = P
        return W, V, I, P

    def convergent_analysis(self, itr=1000, lr=0.03):
        '''
        ダイオードの扱いに関連して収束計算を行う
        ループ内で電圧の移動平均を作り、これに応じてアドミタンスを関数Diode()を用いて調節する．
        これはブランチ自身のアドミタンスでYの対角成分に関する．

        いっぽう、電圧制御電流源の相互コンダクタンスは非対角成分である．
        制御ブランチに逆電圧が印加されたときに無制限に逆方向の電流を生じるのを防ぐ必要がある．
        制御ブランチの電圧が逆方向となった状態は、被制御ブランチに逆接続されるダイオードのpnに
        応じて判定し、
        もとのYに置かれた相互コンダクタンスを電圧が逆方向なら0に、純方向ならば、そのままにする．
        このときループの中では、いったん 0 にされた相互コンダクタンスは、回復できなくなるので、
        判断の都度毎回、元のYをループ外で加工したものを使う．
        ダイオードのように次第に正解に近づける収束計算はいらないからそれでよい．　　        
        
        '''
        A = self.A
        Y = copy.deepcopy(self.Y)
        E = self.E
        pn = self.pn
        V_ppl = [0.0 for e in E]
        
        # 能動素子のための準備 
        eyeY = mm.eye(len(Y))
        nondY = mm.sub(Y, mm.mul(Y, eyeY))      # non-diagonal
        trnsY = mm.mul(mm.trn(pn), nondY)       # transconductanceを符号付きで抽出

        for i in range(itr):
            W, V, I, P = Analize(A, Y, E)

            # 能動素子の逆バイアスの対応(相互コンダクタンス)
            Yjdge = mm.lte(mm.mul(trnsY, V), 0) # 符号付相互コンダクタンス×電圧
            Yvccs = mm.mul(nondY, Yjdge)        # 非対角成分は元の値から作る

            diagY = mm.mul(Y, eyeY)             # 対角成分はループ内で継承         
            Y     = mm.add(diagY, Yvccs)        # 対角非対角をマージ

            # ダイオードの対応(アドミタンス)
            V_ppl = mm.sub(V_ppl, mm.mul(lr, mm.sub(V_ppl, V)))  # 移動平均
            Y = Diode(Y, E, pn, V_ppl)          # ダイオード

        self.W = W
        self.V = V
        self.I = I
        self.P = P
        return W, V, I, P

    def display(self):
        n, b, t, f = self.size
        # Display Nodes
        print()
        for i in range(n):
            print('Node#{:d} potential = {:7.4f}V'.format(i+1, self.W[i]))

        # Display Branches
        print()
        for i in range(b):
            print('branch#{:d}：voltage = {:7.4f}V current = {:7.4f}A power = {:7.4f}W' \
                  .format(i+1, self.V[i], self.I[i], self.P[i]))

        # Display Transistors 
        if t > 0:
            print()
            for k in range(t):
                l = b + 3 * k
                print('Transistor#{:d} Vbe{:7.3f}V Vce{:7.3f}V Ib {:7.3f}mA Ic {:7.3f}mA'\
                      .format(k+1, self.V[l]+self.V[l+2], self.V[l+1]+self.V[l+2], self.I[l]*1000, self.I[l+1]*1000))

        # Display FETs
        if f > 0:
            for k in range(f):
                l = b + 3 * t + 3 * k
                print('FET#{:d}        Vgs{:7.3f}V Vds{:7.3f}V Ids{:7.3f}mA'\
                      .format(k+1, self.V[l]+self.V[l+2], self.V[l+1], self.I[l+1]*1000))
            print()

    def save(self, file_name, title=None):
        # params は辞書形式(export_params で取得したもの)
        params = {}
        params['title'] = title
        params['diode'] = self.diode
        params['size'] = self.size
        params['A'] = self.A
        params['Y'] = self.Y
        params['E'] = self.E
        params['pn'] = self.pn
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print('モデル緒元をファイルに記録しました=>', file_name)    

    def load(self, file_name, dump=False):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        title = params.pop('title', None)
        self.diode = params.pop('diode', False)
        print(title)
        self.size = params['size']
        self.A = params['A'] 
        self.Y = params['Y']
        self.E = params['E']
        self.pn = params['pn']
        print('モデル緒元をファイルから取得しました<=', file_name)
        if dump:
            print(params)
        return self.A, self.Y, self.E, self.pn
        
    def diode2resistance(self):
        n, b, t, f = self.size
        if self.diode:
            for k in range(b): # branches
                if self.pn[k] != 0:
                    self.E[k] += self.pn[k]*0.6
                    print('branch', k+1, 'set', self.E[k])
            for k in range(t): # transistors
                self.Y[b+3*k+1][b+3*k+1] = 1 / 40000 # branch collector to hidden
                self.E[b+3*k+2] = 0.6                # branch hidden to emitter         
            for k in range(f): # FETs
                self.Y[b+3*t+3*k+1][b+3*t+3*k+1] = 1 / 100000 # branch drain to source
        self.pn = [0.0 for i in range(b+3*t+3*f)]
        self.diode=False

    def transient(self, *keys, itr=100, lr=0.03):
        A = self.A
        Y = copy.copy(self.Y)
        E = self.E
        pn = self.pn
        V_ppl = [0.0 for e in E]
        mesure=[[] for k in keys]
        for i in range(itr):
            W, V, I, P = Analize(A, Y, E)
            V_ppl = mm.sub(V_ppl, mm.mul(lr, mm.sub(V_ppl, V)))  # 移動平均
            Y = Diode(Y, E, pn, V_ppl)
            for j, k in enumerate(keys):
                km = k[0]           # 変数名（str）
                idx = int(k[1:])-1  # 変数番号
                ns = locals()       # str->変数
                mesure[j].append(ns[km][idx])
        return mesure     
   

def set_A(n, b):
    '''
    create matrix A
    '''
    A = [[0 for j in range(n)] for i in range(b)]
    print('Input starting node and ending node of each branch.')
    for k in range(b):
        print('Branch#' + str(k+1)) 
        f1 = int(input('the starting node# '))
        if f1 > 0 : # ブランチkの始点　
            A[k][f1 - 1] = 1
        t1 = int(input('the ending node#   '))
        if t1 > 0 : # ブランチkの終点
            A[k][t1 - 1] = -1
    return A

def set_branch(b, diode=False):
    '''
    specify branches
    '''
    Y = [[0.0 for j in range(b)] for i in range(b)]
    E = [0.0 for i in range(b)]
    pn = [0.0 for i in range(b)] 
    print('Input spec of each branch.')
    for k in range(b):
        print('Branch#' + str(k+1))
        if diode:
            Dtype = input('If this is a diode, specify the type [diode/LED] => ')
        else:
            Dtype = None
        if Dtype in ('led', 'LED', 'l', 'L'):
            print('LED')
            pn[k] = 1
            e = 1.4
            y = 0.857
        elif Dtype in ('diode', 'Diode', 'd', 'D'):
            print('diode')
            pn[k] = 1
            e = 0.3
            y = 0.857
        elif Dtype in ('rd', 'RD'):
            print('reverse diode')
            pn[k] = -1
            e = -0.3
            y = 0.857
        else:    
            e = float(input('electro motive force  '))
            r = float(input('electrical resistance '))
            y = 1/r
        Y[k][k] = y
        E[k] = e
        
    if diode:
        return Y, E, pn
    else:
        return Y, E

def set_A_trs(n, t):
    '''
    create matrix A for transistors
    '''
    A = [[0 for j in range(n+t)] for i in range(3*t)]
    print('各Transistorの接続ノードを入力せよ')
    for k in range(t):
        print( str(k+1) + '番目のTransistor')
        base = int(input('Node where Base connected      => '))
        emit = int(input('Node where Emitter connected   => '))
        clct = int(input('Node where Collector connected => '))
        if base > 0:  
            A[3*k][base-1]   =  1
        if clct > 0: 
            A[3*k+1][clct-1] =  1            
        if emit > 0: 
            A[3*k+2][emit-1] = -1
        # hidden node is n+k
        A[3*k][n+k]   = -1 # base to hidden
        A[3*k+1][n+k] = -1 # collector to hidden 
        A[3*k+2][n+k] =  1 # hidden to emitter
    return A

def set_branch_trs(t, diode=False, \
          rb=50, Thc=0.2, rc=40000, The=0.2, ye=0.857, ee=0.6):
    '''
    specify branches of transistors
    '''
    Y = [[0.0 for j in range(3*t)] for i in range(3*t)]
    E = [0.0 for i in range(3*t)]
    pn = [0 for i in range(3*t)] 

    print('各Transistorのデータを入力せよ')
    for k in range(t):
        print( str(k+1) + '番目のTransistor')
        NPN  = -1 if input('Type(NPN/PNP)') in ('p', 'P', 'pnp', 'PNP') else 1
        print('NPN =', NPN)
        hfe  = float(input('current amplification factor (hfe) '))

        # branch base to hidden
        Y[3*k][3*k]         = 1 / rb           # input resistance of base　

        # branch collector to hidden
        if diode:
            pn[3*k+1]       = -NPN             # backward connection
            E[3*k+1]        = Thc*NPN          # threshold 
        else:
            Y[3*k+1][3*k+1] = 1 / rc           # collector resistance
        # Transconductance    
        Y[3*k+1][3*k]       = hfe*Y[3*k][3*k]  # hfe -> Transconductance
            
        # branch hidden to emitter
        if diode:
            pn[3*k+2]       = NPN              # forward connection
            E[3*k+2]        = The*NPN          # threshold
        else:
            E[3*k+2]        = ee*NPN           # voltage offset          
            Y[3*k+2][3*k+2] = ye               # admittance

    if diode:
        return Y, E, pn
    else:
        return Y, E

def set_A_fet(n, t):
    '''
    create matrix A for FETs
    '''
    A = [[0 for j in range(n+t)] for i in range(3*t)]
    print('各FETの接続ノードを入力せよ')
    for k in range(t):
        print( str(k+1) + '番目のFET')
        gt   = int(input('Node where Gate connected      '))
        so   = int(input('Node where Source connected    '))
        dr   = int(input('Node where Drain connected     '))
        if gt > 0:  # ゲート(3*kの始点)
            A[3*k][gt-1]   = 1
        if so > 0:  # ソース(3*kと3*k+1の終点)
            A[3*k+1][so-1] = -1
            A[3*k+2][so-1] = -1
        if dr > 0:  # ドレイン(3*k+1の終点)
            A[3*k+1][dr-1] = 1

        # hidden node is n+k
        A[3*k][n+k]   = -1
        A[3*k+2][n+k] =  1
    return A

def set_branch_fet(t, diode=False,\
                   rg=1000000, Thd=0.5, rd=100000, yhs=0.857):
    '''
    specify branches of FETs
    '''
    Y = [[0.0 for j in range(3*t)] for i in range(3*t)]
    E = [0.0 for i in range(3*t)]
    pn = [0 for i in range(3*t)] 

    print('各FETのデータを入力せよ')
    for k in range(t):
        print( str(k+1) + '番目のFET')
        Nch  = -1 if input('Type(N/P)') in ('p', 'P', 'pch', 'Pch') else 1
        print('Nch =', Nch) 
        gm   = float(input('Transconductance(milli-mho)    '))
        idss = float(input('Drain cut-off Curennt(Idss mA) '))
        vth = idss / gm                   # Vth related to bias of gate

        # branch gate to hidden
        if diode:
            pn[3*k]             = Nch
        else:
            Y[3*k][3*k]         = 1 / rg
       
        # branch drain to source
        if diode:
            pn[3*k+1]       = -Nch        # backward connection
            E[3*k+1]        = Thd*Nch     # threshold regularly-0.6*Nch 
        else:
            Y[3*k+1][3*k+1] = 1 / rd      # drain resistance
        # Transconductance    
        Y[3*k+1][3*k]       = gm / 1000   # transconductance

        # branch hidden to source
        E[3*k+2]        = -vth*Nch        # threshold
        Y[3*k+2][3*k+2] = yhs             # 0.857 but must be less than rg*1000

    if diode:
        return Y, E, pn
    else:
        return Y, E

def set_A_fet4(n, t):
    '''
    create matrix A for FETs
    '''
    A = [[0 for j in range(n+2*t)] for i in range(4*t)]
    print('各FETの接続ノードを入力せよ')
    for k in range(t):
        print( str(k+1) + '番目のFET')
        gt   = int(input('Node where Gate connected      '))
        so   = int(input('Node where Source connected    '))
        dr   = int(input('Node where Drain connected     '))
        if gt > 0:  # ゲート(4*kの始点)
            A[4*k][gt-1]   = 1
        if so > 0:  # ソース(4*k+2と4*k+3の終点)
            A[4*k+2][so-1] = -1
            A[4*k+3][so-1] = -1
        if dr > 0:  # ドレイン(4*k+1の始点)
            A[4*k+1][dr-1] = 1

        # hidden1 node is n+2k
        A[4*k][n+2*k]   = -1
        A[4*k+2][n+2*k] =  1

        # hidden2 node is n+2k+1
        A[4*k+1][n+2*k+1] = -1
        A[4*k+3][n+2*k+1] =  1

    return A

def set_branch_fet4(t, diode=False,\
                   rg=1000000.0, ydd=0.857, Thd=0.3, rd=100000, yh=0.0001):
    '''
    specify branches of FETs
    '''
    Y = [[0.0 for j in range(4*t)] for i in range(4*t)]
    E = [0.0 for i in range(4*t)]
    pn = [0.0 for i in range(4*t)] 

    print('各FETのデータを入力せよ')
    for k in range(t):
        print( str(k+1) + '番目のFET')
        Nch  = -1 if input('Type(N/P)') in ('p', 'P', 'pch', 'Pch') else 1
        print('Nch =', Nch) 
        gm   = float(input('Transconductance(milli-mho)    '))
        idss = float(input('Drain cut-off Curennt(Idss mA) '))
        vth = idss / gm                   # Vth related to bias of gate

        # branch gate to hidden1
        Y[4*k][4*k]     = 1 / rg      # input resistance of gate

        # branch drain to hidden2
        if diode:
            pn[4*k+1]       = -Nch        # backward connection
            E[4*k+1]        = Thd*Nch     # threshold regularly-0.6*Nch 
            Y[4*k+1][4*k+1] = 0.857       # admittance
        else:
            Y[4*k+1][4*k+1] = 1 / rd      # drain resistance
        # Transconductance    
        Y[4*k+1][4*k]       = gm / 1000   # transconductance

        # branch hidden1 to source
        if diode:
            #pn[4*k+2]       = Nch
            E[4*k+2]        = -vth*Nch    # voltage offset
            Y[4*k+2][4*k+2] = 0.857       # 0.857 but < 1000*rg
        else:
            E[4*k+2]        = -vth*Nch    # voltage offset
            Y[4*k+2][4*k+2] = 0.857       # 0.857 but < 1000*rg

        # branch hidden2 to source
        if diode:
            pn[4*k+3]       = Nch
            Y[4*k+3][4*k+3] = 0.857
        else:
            Y[4*k+3][4*k+3] = 0.857

    if diode:
        return Y, E, pn
    else:
        return Y, E

def Analize(A, Y, E):
    '''
    回路解析
    '''
    S = [Y[i][i] * e for i, e in enumerate(E)]
    #S    = mm.dot(mm.mul(mm.eye(len(Y)), Y), E) 
    tA   = mm.trn(A)
    tAS  = mm.dot(tA, S)
    tAY  = mm.dot(tA, Y)        
    tAYA = mm.dot(tAY, A)      
    L    = mm.inv(tAYA)         
    W    = mm.dot(L, tAS)       
    V    = mm.dot(A, W)         
    I    = mm.sub(mm.dot(Y, V), S)        
    P    = mm.mul(mm.sub(V, E), I)
    return W, V, I, P

def diode(v, vt=0.026, i0=3e-9):
    '''
    ダイオードを電圧電流特性からエミュレート
    i = is{exp(vf/vt) - 1} = exp[vf/vt + ln(is)] - is
    Vt = nkT/q = 0.026 [V] 熱起電力
    
    '''
    i = math.exp(v / vt + math.log(i0)) - i0
    return i

def Diode(Y, E, pn, V, Vt=0.026, Is=3e-9): #Is=0.00015
    '''
    ダイオードを電圧電流特性からエミュレート
    I = Is{exp(Vf/Vt) - 1} = exp[Vf/Vt + ln(Is)] - Is
    Vt = nkT/q = 0.026 [V] 熱起電力
    
    '''
    Vf = mm.sub(V, E)
    Vf = mm.mul(Vf, pn)                    # 電圧の向き
    Vf = mm.where(mm.eq(Vf, 0), 1e-12, Vf) # 0除算回避, pn=±1の要素のみ意味がある

    lnIs = mm.log(Is) #math.log(Is)
    X = mm.div(Vf, Vt)                     # 電流はoffsetのついた電圧に対して求める
    Z = mm.add(X, lnIs)
    Z = mm.where(mm.gt(Z, 100), 100, Z)    # オーバーフロー対策 Zが大きいのはおかしい
    Z = mm.exp(Z)
    I = mm.sub(Z, Is)                      # あくまでも順方向を基準に算出  

    vY = mm.div(I, Vf)                     # admittanceは元の電圧に対する値を返す
    vY = mm.where(mm.lt(vY, 1e-12), 1e-12, vY) # 極小負性抵抗値対策
    vY = mm.where(mm.gt(vY, 1e12), 1e12, vY)
    # 形状を揃えて元のYと集約
    mpn = mm.vec2mat(pn)    
    Yd  = mm.vec2mat(vY)    
    Yd  = mm.where(mm.eq(mpn, 0), Y, Yd)

    # Yd >= 0.0
    #if mm.min(Yd) < 0:
    #    print(I, Vg)
    #    raise Exception('Negative resistance', Yd)

    I = mm.mul(I, pn) # 最後に実際の向きに合わせる <= 電流の向きに注意(pn=-1の場合には逆向き)  　　　

    return Yd

def VCCS(Y, pn, V): 
    '''
    電圧制御電流源 Voltage Control Current Source
    自ブランチの電圧に依存してgmを調整
    
    '''
    diagY = mm.mul(Y, mm.eye(len(Y)))       # 対角要素　　　　　
    nondY = mm.sub(Y, diagY)                # 非対角要素
    pck2Y = mm.mul(mm.trn(pn), Y)#; print('pck2Y', pck2Y)
    tgt2Y = mm.mul(pck2Y, V)#; print('tgt2Y', tgt2Y)
    jdg2Y = mm.lte(tgt2Y, 0.0)#; print('jdg2Y', jdg2Y)
    nnd2Y = mm.mul(nondY, jdg2Y)#; print('nnd2Y', nnd2Y)
    rft2Y = mm.add(diagY, nnd2Y)#; print('rft2Y', rft2Y)

    return rft2Y, jdg2Y

def VCCS_sigmoid(Y, V, pol, base=100, offset=-0.2): 
    '''
    電圧制御電流源 Voltage Control Current Source
    自ブランチの電圧に依存してgmを調整
    sigmoid関数を使用
    
    '''
    dependV = mm.mul(V, pol)
    gm_rate = mm.sigmoid(mm.add(dependV, offset), base)
    gm_rate = mm.trn(gm_rate)
    diagonalY = mm.mul(Y, mm.eye(len(V)))
    nondiagonalY = mm.sub(Y, diagonalY)
    nondiagonalY = mm.mul(nondiagonalY, gm_rate)
    return mm.add(diagonalY, nondiagonalY)

def display(W, V, I, P, n, b, t, f):
    # Display Nodes
    print()
    for i in range(n):
        print('Node#{:d} potential = {:7.4f}V'.format(i+1, W[i]))

    # Display Branches
    print()
    for i in range(b):
        print('branch#{:d}：voltage = {:7.4f}V current = {:7.4f}A power = {:7.4f}W' \
              .format(i+1, V[i], I[i], P[i]))

    print()
    # Display Transistors
    for k in range(t):
        l = b + 3 * k
        print('Transistor#{:d} Vbe{:7.3f}V Vce{:7.3f}V Ib {:7.3f}mA Ic {:7.3f}mA'\
              .format(k+1, V[l]+V[l+2], V[l+1]+V[l+2], I[l]*1000, I[l+1]*1000))

    # Display FETs
    for k in range(f):
        l = b + 3 * t + 3 * k
        print('FET#{:d}        Vgs{:7.3f}V Vds{:7.3f}V Ids{:7.3f}mA'\
              .format(k+1, V[l]+V[l+2], V[l+1], I[l+1]*1000))
    print()

