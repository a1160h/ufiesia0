# 2022.05.20 井上
from ufiesia0.Config import *
print(np.__name__, 'is running in', __file__, np.random.rand(1))    

def get_data(rate=None, **kwargs):
    import pickle
    with open('../datasets/sklearn_olivetti_faces', 'rb') as f:
        data   = pickle.load(f)
        target = pickle.load(f)
    correct = np.eye(40)[target]    
    return data, correct, target

def label_list(): # *は女性
    name = 'Jake', 'Tucker', 'Oliver', 'Cooper', 'Duke', 'Buster', 'Buddy', 'Kate*',\
           'Sam', 'Lora*', 'Toby', 'Cody', 'Ben', 'Baxter', 'Oscar', 'Rusty', 'Gizmo', \
           'Ted', 'Murphy', 'Cooper', 'Bentley', 'Wiston', 'William', 'Alex', 'Aaron', \
           'Colin','Daniel','Cooper','Connor','Devin', 'Henry','Sadie*','Ian','James', \
           'Gracie*','Jordan','Joseph','Kevin','Kyle','Luke' 
    print('名簿を提供します')
    return name

def show_sample(data, label=None):
    data = np.array(data.tolist())
    max_picel = np.max(data); min_picel = np.min(data) # 画素データを0～1に補正
    rdata = (data - min_picel)/(max_picel - min_picel)
    rdata = rdata.reshape(64, 64)
    plt.imshow(rdata.tolist(), cmap='gray')
    plt.title(label)
    plt.show()

