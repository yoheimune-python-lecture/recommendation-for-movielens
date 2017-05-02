
# MovieLens を用いたレコメンデーションの実装
このノートでは、レコメンデーションの実装例を示します。

## 1. データの取得
[MovieLens](https://grouplens.org/datasets/movielens/100k/)から利用するデータを取得します。
取得したデータは `data/`ディレクトリに保存し、zip解凍もしておきます。


```python
import os
from urllib.request import urlopen

# MovieLensのサイトから、Zipファイルを取得し、ローカルに保存します.
if not os.path.exists("data/ml-100k.zip"):
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    with urlopen(url) as res:
        with open("data/ml-100k.zip", "wb") as f:
            f.write(res.read())
    # Zipファイルを解凍します.
    from shutil import unpack_archive
    unpack_archive("data/ml-100k.zip", "data/", "zip")
```

## 2. データの前処理
取得したデータのうち `u1.base` という学習用データ（全10万件のうち7万件）を利用します。  
まずは取得したデータをそのままの形で、DataFrameとして読み込んで見ます。


```python
import numpy as np
import pandas as pd
udata = pd.read_csv("data/ml-100k/u1.base", delimiter="\t", names=("user", "movie", "rating", "timestamp"))
udata.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>movie</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79995</th>
      <td>943</td>
      <td>1067</td>
      <td>2</td>
      <td>875501756</td>
    </tr>
    <tr>
      <th>79996</th>
      <td>943</td>
      <td>1074</td>
      <td>4</td>
      <td>888640250</td>
    </tr>
    <tr>
      <th>79997</th>
      <td>943</td>
      <td>1188</td>
      <td>3</td>
      <td>888640250</td>
    </tr>
    <tr>
      <th>79998</th>
      <td>943</td>
      <td>1228</td>
      <td>3</td>
      <td>888640275</td>
    </tr>
    <tr>
      <th>79999</th>
      <td>943</td>
      <td>1330</td>
      <td>3</td>
      <td>888692465</td>
    </tr>
  </tbody>
</table>
</div>



上記の形式だと学習に用いづらいため、 **行が映画、列がユーザーの行列(マトリックス)** に変換します。


```python
data = np.zeros((udata["movie"].max(), udata["user"].max()), dtype=np.int)
data.shape
```




    (1682, 943)



`data.shape` から、映画数=1682、ユーザー数=943のデータであることがわかります。

続いて、上記で作成したマトリックスに、評価データを流し込みます。  
ここでは少しだけ工夫をして、 **評価>=3のみを対象（つまり、好評価のみ）** に、評価データを取り込みます。  
また評価数の情報は消し、評価>=3の場合には「1」を登録することとします。


```python
for i, row in udata.iterrows():
    # ratingが3以上のみを対象にしよう（好評価のみ）
    if row["rating"] >= 3:
        data[row["movie"]-1][row["user"]-1] = 1
```

ついでにDataFrameに変換しておきます。


```python
df = pd.DataFrame(data)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>933</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 943 columns</p>
</div>



（参考までに）  
評価>=3のデータ数を確認してみましょう。


```python
# 評価として取り込んだデータの数
df.astype(bool).sum(axis=1).sum()
```




    66103



全70,000件中、66,103件は好評価のようです（94%）。今回は、評価>=3の考慮はあまり意味がなさそうです(笑)。  
ですが、評価データを扱う場合にはそれがプラス/マイナスのどちらなのかを意識することは重要です。

## 3. 映画同士の類似度を計算する
それでは、学習データからレコメンドデータを作成したいと思います。  

前処理から、DataFrameは「1682 x 943」のデータであることがわかっています（映画数=1682、ユーザー数=943）。
ここでは各映画が943個の特徴を持つベクトルと考えることとし、各ベクトルのコサイン距離から類似度を計算することとします。

まずは簡素化して、以下のような映画が2つあるとします。


```python
item1 = np.array([1,1,0])
item2 = np.array([1,0,1])
```

上記は、それぞれ3つの特徴を持つベクトルで、コサイン距離は以下のように計算します。


```python
from scipy.spatial.distance import cosine
cosine(item1, item2)
```




    0.50000000000000011



ここで、コサイン距離は、ベクトルが直交する（=つまり類似度が全くない）場合には「1」となるため、類似度は以下のように計算します。


```python
sim = 1 - cosine(item1, item2)
print(sim)
```

    0.5


上記の要領で、映画ID=1と映画ID=2の類似度を計算してみると、以下のようになります。


```python
sim = 1 - cosine(data[0], data[1])
print(sim)
```

    0.325207858278


上記の `0.32`は相対的な数値でありそれ自体に意味はありませんが、他の類似度と比較することで、より類似しているアイテムを見つけることができます。

上記の感じを、総当たりで全アイテムの類似度を計算します。  
ここではscipyの `pdist` を用いてお手軽に行います。


```python
# 上記の雰囲気で、総当たりで全アイテムの距離を計算する.
from scipy.spatial.distance import pdist
d = pdist(data, "cosine")
# 類似度 = 1 - コサイン距離
d = 1 - d

# 結果を行列に変換します（上記だとベクトルで見辛い！！）
from scipy.spatial.distance import squareform
d = squareform(d)
# nan ができるので、0に補正します.
d[np.isnan(d)] = 0

# ここでちょっとしたトリックで、自分自身は「-1」に補正して、類似度を最低にします.
d = d - np.eye(d.shape[0])

# 表示してみる.
print(d)
```

    [[-1.          0.32520786  0.27172635 ...,  0.          0.05322463
       0.05322463]
     [ 0.32520786 -1.          0.20689728 ...,  0.          0.10910895
       0.10910895]
     [ 0.27172635  0.20689728 -1.         ...,  0.          0.          0.14586499]
     ..., 
     [ 0.          0.          0.         ..., -1.          0.          0.        ]
     [ 0.05322463  0.10910895  0.         ...,  0.         -1.          0.        ]
     [ 0.05322463  0.10910895  0.14586499 ...,  0.          0.         -1.        ]]


上記で、各映画ごとの類似度を総当たりで計算することができました。  
この後のレコメンドの作成は、上記のデータを用いて推薦するアイテムを作成します。

## 4. レコメンドデータを作成する

例えば、映画ID=1に類似する映画を、類似度の高い順に並べてみます。


```python
movie_id = 0  # DataFrameのindexは 0 始まり

#  映画IDと評価のマップを作成します.
recommends = {}
for index, sim in enumerate(d[movie_id]):
    if sim > 0:
        recommends[index] = sim

# 評価の良い順に並べます.
recommends = sorted(list(recommends.items()), key=lambda r:r[1], reverse=True)

# 最初の10件を表示してみます.
from pprint import pprint
pprint(recommends[:10])
```

    [(49, 0.62828380959743046),
     (180, 0.60179361066116654),
     (120, 0.56361171748207761),
     (116, 0.55909527338435139),
     (221, 0.5448072284259402),
     (404, 0.53934291268451406),
     (256, 0.52993874949054487),
     (236, 0.52990539045000284),
     (6, 0.52766307860331085),
     (150, 0.51909505604442208)]


映画IDが「49, 180, 120, ・・・」と類似度の高い順に並んでいることがわかります。  

さて、上記の処理を応用して、指定したユーザーへ映画を10本レコメンドする関数を実装します。処理の流れは以下の通りです。  


**指定したユーザーへ映画を10本レコメンドする関数の仕様**
* 指定されたユーザーが評価した映画一覧を取得する
* 各映画に対してレコメンド候補を取得する（上の映画ID=1の類似作品の処理と同じ）
* レコメンド候補から、すでに閲覧済のデータは除去します
* レコメンド候補から、上位10件を返却します

具体的には以下のような実装となります。


```python
# 指定したユーザーへレコメンドするアイテムを10個出力する関数
def get_recommend_items(user_id):
    # 指定ユーザーが評価した映画一覧を取得.
    used = set(df[user_id].nonzero()[0].tolist())
    # レコメンドを作成.
    candidates = {}
    for movie_id in used:
        for index, sim in enumerate(d[movie_id]):
            if sim > 0:
                candidates[index] = sim
    candidates = sorted(list(candidates.items()), key=lambda r:r[1], reverse=True)
    # すでに閲覧済は除く.
    recommends = []
    for c in candidates:
        if c[0] not in used:
            recommends.append(c)

    # 返却.
    return [r[0] for r in recommends[:10]]

# 試しにUser_ID=100の人
recommends = get_recommend_items(100)
print(recommends)
```

    [236, 110, 297, 49, 120, 281, 470, 814, 14, 404]


これで、レコメンド処理の実装が完了しました！！！

## 5. レコメンド結果の評価
上記で作成したレコメンドについて、どれほどの良いのか（悪いのか）評価したいと思います。  
ここでは評価用のデータ（u1.test）を用いて評価を行います。

**[評価方法]**
* 評価データにあるユーザーに対して、レコメンドを10件表示する
  * レコメンドの生成は、上記で定義した「**get_recommend_items**」を用います。
* 表示したレコメンド10件のうち、1件以上、評価データ内で閲覧したデータがあれば成功。
* 「成功数 / ユーザー数」で精度を測る。

まずはテストデータを読み込みます。


```python
utest = pd.read_csv("data/ml-100k/u1.test", delimiter="\t", names=("user", "movie", "rating", "timestamp"))
utest.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>movie</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19995</th>
      <td>458</td>
      <td>648</td>
      <td>4</td>
      <td>886395899</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>458</td>
      <td>1101</td>
      <td>4</td>
      <td>886397931</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>459</td>
      <td>934</td>
      <td>3</td>
      <td>879563639</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>460</td>
      <td>10</td>
      <td>3</td>
      <td>882912371</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>462</td>
      <td>682</td>
      <td>5</td>
      <td>886365231</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 行が映画、列がユーザーのマトリックスを作成.
test = np.zeros((utest["movie"].max(), utest["user"].max()), dtype=np.int)
test.shape
```




    (1591, 462)




```python
# 上記で作成したマトリックスに、データを流し込む.
for i, row in utest.iterrows():
    # ratingが3以上のみを対象にしよう（好評価のみ）
    if row["rating"] >= 3:
        test[row["movie"]-1][row["user"]-1] = 1
```


```python
df_test = pd.DataFrame(test)
df_test.shape
```




    (1591, 462)



と、ここまでで評価用のDataFrameを作成することができました。  

試しに、userId=1の人で、レコメンドが成功するかをテストしてみたいと思います。


```python
# 試しに、userId=1の人でテスト.
used = set(df_test[0].nonzero()[0].tolist())
recommends = set(get_recommend_items(0))
used & recommends
```




    {257, 271}



無事にレコメンドができたようです（ホッとしますw）。

続けて、他の人も評価を行なっていきましょう。


```python
# 続けて他の人もやってみよう.
all = 0
good = 0
for user_id in range(df_test.shape[1]):
    used = set(df_test[user_id].nonzero()[0].tolist())
    recommends = set(get_recommend_items(user_id))
    items = used & recommends
    good += (1 if items else 0)
    all += 1

print("全件={0}, 成功数={1}, 成功率={2}%".format(all, good, good * 100 // all))
```

    全件=462, 成功数=334, 成功率=72%


今回の場合には、72%の確率が、ユーザーが興味を持つかもしれない映画をレコメンドすることができました。 

めでたしめでたし。


```python

```
