import random
from janome.tokenizer import Tokenizer

def split_by_janome(text):
    """ janomeの形態素分析で文章を語に分割 """
    text = text.replace('\n', '') #改行削除
    text = text.replace('\r', '') #スペース削除
    word_list = Tokenizer().tokenize(text, wakati=True)
    return word_list

  
def make_markov(x_chain, n=2, verbose=False):
    """ マルコフ連鎖のテーブル作成(キーはn個の連続した値)
    P(Xt+1 | Xt=At, Xt-1=At-1, --- )
      x      key2   key1       ---
    """
    markov = {}
    key = []    # キーはリストで操作して辞書アクセスの際にタプルにする
    for x in x_chain:
        # -- 1つ前のxでキーの準備ができていたら実行 --
        if len(key) == n:
            k = tuple(key)
            # 該キーに対応するものがなければ空のリストで用意
            if k not in markov:
                markov[k] = []
                if verbose:
                    print('create key', k)
            # 該キーのリスト中に対象がなかったら加える
            if x not in markov[k]:
                markov[k].append(x)
                if verbose:
                    print('append key', k, ':', x)
        # -- 次のxに向けてキーを整える --            
        key.append(x) 
        if len(key) > n:
            key = key[1:] # 左端を捨てて左シフト

    return markov    
  
def generate_text(markov, length=100, start=None, end=None, print_text=True):
    """ マルコフ連鎖のテーブルを使って自動生成 """
    count = 0
    x_chain = "" # 空の文字列
    
    # -- 所定の長さのstartが指定されたらキーとし、そうでなければランダムに選んだキー
    key = random.choice(list(markov.keys()))
    n = len(key)
    if start is not None and len(start)==n: # 長さがあっている時だけ置き換える
        key = start
    key = list(key)

    # -- 出だしはキーそのもの --
    if print_text:    
        for k in key:
            print(k, end='')
    
    while count < length:
        # -- キーの指すものからランダムに選んで綴っていく --
        k = tuple(key)
        x = random.choice(markov[k])
        x_chain += x
        count += 1
        if print_text:    
            print(x, end='')
        if x==end:
            break
        # -- 次に向けてキーを左シフトする --    
        key.append(x)
        key = key[1:]
           
    return x_chain
     
