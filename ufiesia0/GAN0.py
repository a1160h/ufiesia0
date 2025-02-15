# GAN
# 2022.06.22 井上
from ufiesia0.Config import *
from ufiesia0 import NN, LossFunctions

class GAN_m_m:
    def __init__(self, nz, out, **kwargs):
        opt_for_gen, opt_for_dsc = {}, {}
        keys_kwargs = list(kwargs.keys()) # iterator(ループ内変更不可)
        for k in keys_kwargs: 
            if k[:4]=='gen_':
                opt_for_gen[k[4:]]=kwargs.pop(k) 
            if k[:4]=='dsc_':
                opt_for_dsc[k[4:]]=kwargs.pop(k)
        # 生成器のオプション        
        opt_for_gen['ml_nn']  = opt_for_gen.pop('ml_nn',       256 )
        opt_for_gen['ml_act'] = opt_for_gen.pop('ml_act',    'ReLU')
        opt_for_gen['ol_act'] = opt_for_gen.pop('ol_act',    'Tanh')
        opt_for_gen['width']  = opt_for_gen.pop('width',      0.01 )
        
        # 判別器のオプション　
        opt_for_dsc['ml_nn']  = opt_for_dsc.pop('ml_nn',       256 )
        opt_for_dsc['ml_act'] = opt_for_dsc.pop('ml_act',   'LReLU')
        opt_for_dsc['ol_act'] = opt_for_dsc.pop('ol_act', 'Sigmoid')
        opt_for_dsc['width']  = opt_for_dsc.pop('width',      0.01 )

        # 生成機と判別器
        print(opt_for_gen, opt_for_dsc)
        self.gen = NN.NN_m(out, **opt_for_gen)
        self.dsc = NN.NN_m(1,   **opt_for_dsc)
        self.loss_function = LossFunctions.CrossEntropyError2()
        self.nz = nz # ノイズの大きさ
        self.gen.summary()
        self.dsc.summary()
        print('loss_function =', self.loss_function.__class__.__name__, '\n')

    def get_accuracy(self, y, t):
        correct = np.sum(np.where(y<0.5, 0, 1) == t)
        return correct / len(y)

    def train_discriminator(self, data, batch_size, eta):
        n_itr = len(data)//batch_size
        valid = np.ones((batch_size, 1))
        fake  = np.zeros((batch_size, 1))
        err_true, acc_true, err_fake, acc_fake = 0, 0, 0, 0
        rand = np.random.randint(0, len(data), len(data))
        for i in range(n_itr):
            # -- 本物画像で判別器の訓練
            idx = rand[i*batch_size:(i+1)*batch_size]
            imgs_real = data[idx]
            y = self.dsc.forward(imgs_real)
            err_true += self.loss_function.forward(y, valid)
            acc_true += self.get_accuracy(y, valid)
            gy = self.loss_function.backward()        
            self.dsc.backward(gy)
            self.dsc.update(eta=eta)

        #for i in range(n_itr):
            # -- 生成画像で判別器の訓練
            noise = np.random.randn(batch_size, self.nz)       
            imgs_fake = self.gen.forward(noise)
            y = self.dsc.forward(imgs_fake)
            err_fake += self.loss_function.forward(y, fake)
            acc_fake += self.get_accuracy(y, fake)
            gy = self.loss_function.backward()        
            self.dsc.backward(gy)
            self.dsc.update(eta=eta)

        return float(err_true/n_itr), float(acc_true/n_itr), \
               float(err_fake/n_itr), float(acc_fake/n_itr)
       
    def train_generator(self, data, batch_size, eta):
        n_itr = len(data)//batch_size #* 2 
        err_trick, acc_trick = 0, 0 
        for i in range(n_itr):
            # -- 判別器を騙して生成器を訓練
            t = np.ones((batch_size, 1))
            noise = np.random.randn(batch_size, self.nz)       
            imgs_fake = self.gen.forward(noise)
            y = self.dsc.forward(imgs_fake)
            self.loss_function.forward(y, t)
            acc_trick += self.get_accuracy(y, 0)
            gy = self.loss_function.backward()
            gx = self.dsc.backward(gy)
            gx = self.gen.backward(gx)
            self.gen.update(eta=eta)
            err_trick += self.loss_function.forward(y, 0)

        return float(err_trick/n_itr), float(acc_trick/n_itr)

    def train(self, data, batch_size, eta):
        n_itr = len(data)//batch_size
        valid = np.ones((batch_size, 1))
        fake  = np.zeros((batch_size, 1))
        err_true, acc_true, err_fake, acc_fake, err_trick, acc_trick \
                  = 0, 0, 0, 0, 0, 0
        rand = np.random.randint(0, len(data), len(data))
        for i in range(n_itr):
            # -- 本物画像で判別器の訓練
            idx = rand[i*batch_size:(i+1)*batch_size]
            imgs_real = data[idx]
            y = self.dsc.forward(imgs_real)
            err_true += self.loss_function.forward(y, valid)
            acc_true += self.get_accuracy(y, valid)
            gy = self.loss_function.backward()        
            self.dsc.backward(gy)
            self.dsc.update(eta=eta)
            # -- 判別器を騙して生成器を訓練
            noise = np.random.randn(batch_size, self.nz)       
            # -- 判別機に本物と偽って騙し生成器を訓練
            imgs_fake = self.gen.forward(noise)
            y = self.dsc.forward(imgs_fake)
            err_trick += self.loss_function.forward(y, valid)
            acc_trick += self.get_accuracy(y, fake)
            gy = self.loss_function.backward()
            gx = self.dsc.backward(gy)
            gx = self.gen.backward(gx)
            self.gen.update(eta=eta)
            # -- 騙すのに使った生成画像について判別器を修正
            imgs_fake = self.gen.forward(noise)
            y = self.dsc.forward(imgs_fake)
            err_fake += self.loss_function.forward(y, fake)
            acc_fake += self.get_accuracy(y, fake)
            gy = self.loss_function.backward()
            self.dsc.backward(gy)
            self.dsc.update(eta=eta)


        return float(err_true/n_itr),  float(acc_true/n_itr), \
               float(err_fake/n_itr),  float(acc_fake/n_itr), \
               float(err_trick/n_itr), float(acc_trick/n_itr)

