# QLearnAgent
# Q学習のエージェント
# 20230114 井上

import sys
import pickle
from ufiesia0.Config import *
from ufiesia0 import NN
from ufiesia0 import RNN

class BaseAgent:
    def __init__(self, **kwargs):
        # パラメータ
        self.epsilon  = kwargs.pop('epsilon',   0.1) # イプシロングリーディ法の乱雑度
        self.gamma    = kwargs.pop('gamma',     0.9) # 期待値算出の割引率
        self.filepath = kwargs.pop('filepath', None)
        self.n_act    = kwargs.get('n_act',       2) # 動作の種類,Qテーブルに必要だから保存
        self.alpha    = kwargs.pop('alpha',     0.1)

    def select_action(self, obs, epsilon=None):
        """ イプシロングリーディ法で観測に対して行動を決定 """
        # epsilonを実行時に指定することも出来る
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:             # epsilon の確率            
            act = np.random.randint(0, self.n_act) # ランダム行動
            #print('random', end=':')
        else:                                      # 1-epsilon の確率
            # -- Qが最大の行動のうちのいずれかを選ぶ --
            Q = self.get_Q(obs)                    # Qを取得
            maxQ = np.max(Q)                       # Qの最大値
            maxi, = np.where(Q==maxQ)              # 最大値と一致するインデクス(１つか複数か)
            act = np.random.choice(maxi)           # その中から１つランダムに選ぶ
            #print('max_Q ', end =':')
        return act

    def learn(self, obs, act, rwd, done, next_obs, is_learn=False):
        """ 学習 """
        #print('<1>', obs, act, rwd, done, next_obs)
        if rwd is None or not is_learn:
            return

        if hasattr(self, 'replay_memory'): # 経験再生メモリがある場合
            self.replay_memory.add(obs, act, rwd, done, next_obs)
            if len(self.replay_memory) < self.replay_memory.batch_size:
                return
            obs, act, rwd, done, next_obs \
                  = self.replay_memory.sample()
        
        target = self._get_target(rwd, done, next_obs)      
        #print('<2>', obs, act, rwd, done, next_obs)
        self._model_update(obs, act, done, target)

    def _get_target(self, rwd, done, next_obs):
        """ ターゲット作成 最終状態ならrwdだけで、さもなくば次の最大のを加味して """
        next_Q = self.get_Q(next_obs)
        target = rwd + (1 - done) * self.gamma * np.max(next_Q)
        return target        

    def show_Q(self, obss):
        """ Q値の表示 """
        if obss is None:
            return
        for obs in obss:
            obs = np.array(obs)
            q_vals = self.get_Q(obs)
            if q_vals is not None:
                valstr = [f' {v: .2f}' for v in q_vals]
                valstr = ','.join(valstr)
                print('{}:{}'.format(str(obs), valstr))

    def get_Q(self, **kwargs):
        raise NotImplementedError()

    def _model_update(self, **kwargs):
        raise NotImplementedError()

    def save_weights(self, **kwargs):
        raise NotImplementedError()
        
    def load_weights(self, **kwargs):
        raise NotImplementedError()

class TableQAgt(BaseAgent):
    """ Qテーブルを使ったQ学習エージェントクラス """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ モデル＝Qテーブルの構築 """
        self.init_val_Q = kwargs.pop('init_val_Q',   0)
        self.max_memory = kwargs.pop('max_memory', 500)
        self.Q = {}     # Qテーブル

    def get_Q(self, obs):
        """ 観測に対するQ値を出力 """
        obs = str(obs)
        self._check_and_add_observation(obs)
        val = self.Q[obs]
        return val #np.array(val)

    def _model_update(self, obs, act, done, target):
        # doneは使わない（インターフェース互換のため）
        obs = str(obs)
        self._check_and_add_observation(obs)
        alpha = self.alpha
        self.Q[obs][act] = (1 - alpha) * self.Q[obs][act] + alpha * target

    def _check_and_add_observation(self, obs):
        """ obs が登録されていなかったら初期値を与えて登録 """
        if obs not in self.Q:
            self.Q[obs] = [self.init_val_Q] * self.n_act  
            len_Q = len(self.Q)
            if len_Q > self.max_memory:  
                print(f'観測の登録数が上限 {self.max_memory:d} に達しました。')
                sys.exit()
            if (len_Q < 100 and len_Q % 10 == 0) or (len_Q % 100 == 0):  
                print(f'the number of obs in Q-table --- {len_Q:d}')

    def display_tableQ(self):
        """ Qテーブルの表示(TableQAgt専用) """
        tableQ = self.model.Q
        if len(tableQ)==0:
            print('tableQ is empty')
        for obs in tableQ.keys():
            q_vals = tableQ[obs]
            if q_vals is not None:
                print('{}: {: .2f}, {: .2f}'\
                           .format(obs, tableQ[obs][0], tableQ[obs][1]))
            else:
                print('{}:'.format(obs))

    def save_weights(self, filepath=None):
        """ モデルの重みデータの保存 """
        if filepath is None:
            filepath = self.filepath
        with open(filepath+'.pkl', mode='wb') as f:
            pickle.dump(self.Q, f)
        
    def load_weights(self, filepath=None):
        """ モデルの重みデータの読み込み """
        if filepath is None:
            filepath = self.filepath
        with open(filepath+'.pkl', mode='rb') as f:
            self.Q = pickle.load(f)

class NetQAgt(BaseAgent):
    """ Qネットワークを使ったQ学習エージェントクラス """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self._build_model(**kwargs)
        
    def _build_model(self, **kwargs):
        """ 指定したパラメータでQネットワークを構築 """
        self.n_dense    = kwargs.pop('n_dence',  128)
        model = NN.NN_mm(
            self.n_act,                  # 出力数
            ml1_nn = self.n_dense,        # 中間層のニューロン数
            ml2_nn = self.n_dense,        # 中間層のニューロン数
            ml1_act = 'ReLU',             # 中間層の活性化関数
            ml2_act = 'ReLU',             # 中間層の活性化関数
            ol_act = 'Identity',         # 出力層の活性化関数
            loss_f = 'MeanSquaredError', # 損失関数
            optimize = 'Adam'
            )
        model.summary()
        return model

    def get_Q(self, obs):
        obs = np.array(obs)
        obs = obs.reshape(1, -1) 
        Q = self.model.forward(obs)
        Q = Q[0, :]
        return Q

    def _model_update(self, obs, act, done, target):
        """ モデルの更新、バッチ処理にも対応(actの数で判別) """
        # doneは使わない（インターフェース互換のため）
        if isinstance(act, int) or isinstance(act, np.int64):
            act = [act]
        # 出力にターゲットを組み込む        
        obs = obs.reshape(len(act), -1)
        y = self.model.forward(obs)
        t = y.copy()          # tの操作でyが変わらないように
        idx = range(len(act)) 
        t[idx, act] = target  # actに対してのみtargetをセットし他はそのまま 
        # モデルの更新
        loss = self.model.loss_function.forward(y, t)
        self.model.backward()
        self.model.update(eta=self.alpha)          

    def save_weights(self, filepath=None):
        """ モデルの重みデータの保存 """
        if filepath is None:
            filepath = self.filepath
        self.model.save_parameters(filepath)
        
    def load_weights(self, filepath=None):
        """ モデルの重みデータの読み込み """
        if filepath is None:
            filepath = self.filepath
        self.model.load_parameters(filepath)

        
class ReplayMemory:  
    """ 経験を記録するクラス """
    def __init__(self, **kwargs):
        self.memory_size = kwargs.pop('replay_memory_size', 10000)
        self.batch_size  = kwargs.pop('replay_batch_size',     32)
        self.items = None              # 登録項目数(初回登録時に確定)
        self.index = None              # 登録番号
        
    def __len__(self):  
        """ len()で、memoryの長さを返す """
        return len(self.memory[0])

    def add(self, *experience):  
        """ 経験を記憶にnumpy配列で追加する """
        # 初回登録 
        if self.items is None: 
            self.items = len(experience)
            self.memory = [[] for i in experience]
            for i, exp in enumerate(experience):
                if type(exp).__module__ == np.__name__:
                    self.memory[i] = exp.reshape((1,) + exp.shape)
                else:
                    self.memory[i] = np.array(exp).reshape(-1)

        # 2回目以降の登録            
        else:
            for i, exp in enumerate(experience):
                if self.memory[i].ndim >= 2:
                    exp = exp.reshape((1,) + exp.shape)
                    self.memory[i] = np.vstack((self.memory[i], exp))
                else:
                    self.memory[i] = np.hstack((self.memory[i], exp))

        # memory_size以下では指標範囲は登録数
        if len(self.memory[0]) <= self.memory_size:
            self.index = np.arange(len(self.memory[0]))
            return

        # memory_sizeを超えたら一番古いものを捨てる
        for i in range(self.items):
            self.memory[i] = self.memory[i][1:]
        
    def sample(self):#, data_length):  
        """ batch_size分、ランダムにサンプルする """
        np.random.shuffle(self.index)
        idx = self.index[:self.batch_size]
        out = []
        for i in range(self.items):
            out.append(self.memory[i][idx])
        return out

class ReplayQAgt(NetQAgt):
    """ 経験再生とQネットワークを使ったQ学習エージェントクラス """
    def __init__(self, **kwargs):
        # 継承元のクラスの初期化
        super().__init__(**kwargs)
        # 経験再生メモリの初期化
        self.replay_memory = ReplayMemory(**kwargs)

    def _get_target(self, rwds, dones, next_obss):
        """ ターゲット作成:バッチ処理 ~で真偽反転 """
        next_obss = next_obss.reshape(len(rwds), -1)
        next_ys = self.model.forward(next_obss)
        targets = rwds + ~dones * self.gamma * np.max(next_ys, axis=-1)
        return targets

class TargetQAgt(NetQAgt):
    """ 経験再生にターゲットネットワークを取り入れたQ学習エージェントクラス """
    def __init__(self, **kwargs):
        # 継承元のクラスの初期化
        super().__init__(**kwargs)
        # ターゲットモデルの初期化
        self.model_target = self._build_model(**kwargs)
        self.time = 0  
        self.target_interval = kwargs.pop('target_interval', 20)
        # 経験再生メモリの初期化
        self.replay_memory = ReplayMemory(**kwargs)

    def _get_target(self, rwds, dones, next_obss):
        """ ターゲットモデルでターゲットを作成 """
        next_obss = next_obss.reshape(len(rwds), -1)
        next_zs = self.model_target.forward(next_obss)
        targets = rwds + ~dones * self.gamma * np.max(next_zs, axis=-1)
        return targets

    def _model_update(self, obs, act, done, target):
        super()._model_update(obs, act, done, target)
        # -- ターゲットモデルの更新 --
        if self.time % self.target_interval == 0 and self.time > 0:
            params = self.model.export_params()
            self.model_target.import_params(params)
        self.time += 1
        

class DoubleQAgt(NetQAgt):
    """ 経験再生にターゲットネットワークを取り入れたQ学習エージェントクラス """
    def __init__(self, **kwargs):
        # 継承元のクラスの初期化
        super().__init__(**kwargs)
        # ターゲットモデルの初期化
        self.model_target = self._build_model(**kwargs)
        self.time = 0  
        self.target_interval = kwargs.pop('target_interval', 20)
        # 経験再生メモリの初期化
        self.replay_memory = ReplayMemory(**kwargs)

    def _get_target(self, rwds, dones, next_obss):
        """ 自身の最大のQを得るインデクスでターゲットモデルの次の一手を選ぶ """
        # 両モデルからターゲット作成:バッチ処理 ~で真偽反転
        next_obss = next_obss.reshape(len(rwds), -1)
        next_zs = self.model_target.forward(next_obss) 
        next_ys = self.model.forward(next_obss)
        # モデルQ値の最大値インデクス
        raw = range(len(rwds))
        column = np.argmax(next_ys, axis=-1)
        # 上記インデクスに対応するターゲットモデルQ値
        targets = rwds + ~dones * self.gamma * next_zs[raw, column]
        return targets

    def _model_update(self, obs, act, done, target):
        super()._model_update(obs, act, done, target)
        # -- ターゲットモデルの更新 --
        if self.time % self.target_interval == 0 and self.time > 0:
            params = self.model.export_params()
            self.model_target.import_params(params)
        self.time += 1


class RNNQAgt(NetQAgt):
    """ Qネットワークを使ったQ学習エージェントクラス """

    def _build_model(self, **kwargs):
        """ 指定したパラメータでQネットワークを構築 """
        # Qネットワークの構築 (A)
        self.input_size = kwargs.pop('input_size',   4)
        self.n_dense    = kwargs.pop('n_dence',     64)
        self.alpha      = kwargs.pop('alpha',     0.03)
        
        model = RNN.RNN_for_Agent(
            self.input_size,             # 入力数
            self.n_dense,                # 隠れ層ニューロン数
            self.n_act,                  # 出力数
            optimize='Momentum'
            )

        self.count = 0
        return model

    def get_Q(self, obs):
        #obs = obs/3
        obs = obs.reshape(-1)
        Q = self.model.step_only(obs)
        return Q.reshape(-1)

    def _model_update(self, obs, act, done, target):
        # 出力にターゲットを組み込む
        Q = self.get_Q(obs) 
        t = Q.copy()       # tの操作でyが変わらないように
        t[act] = target    # actに対してのみtargetをセットし他はそのまま 

        # モデルの更新：step_and_stackで蓄積しておいて、doneで逆伝播して更新する
        #obs = obs/3
        """
        obs = obs.reshape(-1)
        self.model.step_and_stack(obs, t)
        self.count += 1
        if done and self.count>1:
        #if self.count > 25:
            self.model.reflect(eta=self.alpha)#, g_clip=50)
            self.model.reset_state()
            #print(self.count)
            self.count = 0
        """
        obs = obs.reshape(1, 1, -1)
        t = t.reshape(1, 1, -1)
        self.model.forward(obs, t)
        self.model.backward()
        self.model.update(eta=self.alpha, g_clip=0.5)
        #"""#

