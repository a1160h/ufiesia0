class Config:
    np    = None
    dtype = 'f4'
    seed  = None

def set_dtype(value):
    #print('old_value =', getattr(Config, 'dtype'))
    setattr(Config, 'dtype', value)
    print('Config.dtype is set to', Config.dtype)

def set_seed(value):
    #print('old_value =', getattr(Config, 'seed'))
    setattr(Config, 'seed', value)    
    np.random.seed(seed=Config.seed)
    print('random.seed', Config.seed, 'is set for', np.__name__)

def set_np(value=None):
    global np
    #print('Config.np old_value =', getattr(Config, 'np'))

    if value is None:
        try:
            import cupy as np
        except:
            import numpy as np
    elif value == 'numpy':
        import numpy as np
    elif value == 'cupy':
        import cupy as np
    else:
        raise Exception("Invalid library specified. Specify either 'numpy' or 'cupy'.")

    if np.__name__ == 'numpy':
        np.seterr(divide='raise') # 割算例外でnanで続行せずに例外処理させる
        #np.seterr(over='raise')

    setattr(Config, 'np', np)    

set_np()    

print(np.__name__, 'is running in', __file__, np.random.rand(1))
print('Config.dtype =', Config.dtype)
print('Config.seed =', Config.seed)
print("If you want to change np, run 'set_np('numpy' or 'cupy'); np = Config.np.'")
print("If you want to change Config.dtype, run 'set_dtype('value')'")
print("If you want to set seed for np.random, run set_seed(number)")

