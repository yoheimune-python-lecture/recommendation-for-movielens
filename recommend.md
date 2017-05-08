
# MovieLens を用いたレコメンデーションの実装サンプル
ここでは、レコメンデーションの実装例を示します。

## 1. データの取得
[MovieLens](https://grouplens.org/datasets/movielens/100k/) から利用するデータを取得します。取得したデータは **data/** ディレクトリに保存し、zip解凍を行います。


```python
import os
from urllib.request import urlopen

# MovieLensのサイトから、Zipファイルを取得し、ローカルに保存します.
# この処理は少しだけ時間がかかるので、未ダウンロードの場合のみ、実行します.
file_name = "data/ml-100k.zip"
if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))
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
まずは、取得したデータをそのままの形で、DataFrameとして読み込んでみます。


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



上記の形式だとモデルの学習に用いづらいため、 **行が映画、列がユーザーの行列(マトリックス)** に変換します。  
ここでは `pivot` メソッドを使用します。

そして今回は、少しだけ工夫をして、 **評価>=3のみ（つまり好評価のみ）を対象** に、評価データを取り込みます。  

また評価数の情報は消し、評価>=3の場合には「1」を登録することとします。  
*（評価をそのまま使うのか、「1」などにマスキングするのかは、精度が良い方にすればOKです）*


```python
# 評価が3以上のデータを抽出.
high_rate = udata.loc[udata["rating"] >= 3]
# movieを行, columnsを列にした後、欠損部分（=NaN）を0埋め.
raw = high_rate.pivot(index="movie", columns="user", values="rating")
df = raw.fillna(0)
# whereメソッドはわかりにくいですが、以下で3未満以外(つまり3以上)を1で埋めて返します
df = df.where(df < 3, 1)
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
      <th>943</th>
    </tr>
    <tr>
      <th>movie</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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



全70,000件中、66,103件は好評価のようです（94%）。今回は、評価>=3の考慮はあまり意味がないかもしれません(笑)。  
ですが、評価データを扱う場合にはそれがプラス/マイナスのどちらなのかを意識することは重要です。

## 3. 映画同士の類似度を計算する
それでは、学習データからレコメンドモデルを作成したいと思います。  

前処理から、DataFrameは「1682 x 943」のデータです（映画数=1682、ユーザー数=943）。  

そしてここでは各映画を、943個の特徴を持つベクトルと考えることにしましょう。  

この時、2つのベクトル（=各映画）の近さ（=類似度）をどのように表現すれば良いでしょうか？  
様々な方法がありますがここでは、2つのベクトルのなす角のコサインの値（=コサイン距離）を類似度として考えます。2つのベクトルが重なり合っている（なす角が0度）の場合にはコサイン=1で類似度Max、2つのベクトルが直行する場合にはコサイン=0で類似度0という具合です。

まずは簡素化して、以下のような映画が2つあるとします。


```python
item1 = np.array([1,1,0])
item2 = np.array([1,0,1])
```

上記は、それぞれ3つの特徴を持つベクトルで、コサイン距離（=類似度）は以下のように計算します。


```python
from scipy.spatial.distance import cosine
sim = 1 - cosine(item1, item2)
print(sim)
```

    0.5


上記の要領で、実際に映画ID=1と映画ID=2の類似度を計算してみると、以下のようになります。


```python
sim = 1 - cosine(df.iloc[0], df.iloc[1])
print(sim)
```

    0.325207858278


上記の `0.32`は相対的な数値でありそれ自体に意味はありませんが、他の類似度と比較することで、より類似しているアイテムを見つけることができます。

上記の雰囲気で、総当たりに全アイテムの類似度を計算します。  
ここでは scipy の `pdist` を用いてお手軽に行います。


```python
# 上記の雰囲気で、総当たりで全アイテムの距離を計算する.
from scipy.spatial.distance import pdist

# 類似度
d = pdist(df, "cosine")
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
この類似度表を用いて、推薦するアイテムを作成します。

## 4. レコメンドデータを作成する

例えば、映画ID=1に類似する映画を、類似度の高い順に並べてみます。  

メモリ効率や速度を考え、Numpyを使います。


```python
# 映画ID=1（indexが0始まりなことに注意）
movie_id = 0

# 評価の良い順に並べます.
# ソート後のインデックスを収納.
id = d[movie_id].argsort()[::-1]

# 最初の5件を表示してみます.
for i in id[:5]:
    print("{i:0>3d}: {v: .3f}".format(i=i, v=d[movie_id, i]))
```

    049:  0.628
    180:  0.602
    120:  0.564
    116:  0.559
    221:  0.545


上記の処理では、指定した映画に類似する映画を知ることができます。  
この実装を応用して、指定したユーザーへ映画を5本レコメンドする関数を実装します。処理の流れは以下の通りです。  

**指定したユーザーへ映画を5本レコメンドする関数の仕様**
* 指定されたユーザーが評価した映画一覧を、学習用データから取得する
* 各映画に対してレコメンド候補を取得する（**上記の処理がこちら**）
* レコメンド候補から、すでに閲覧済のデータは除去する
* レコメンド候補から、上位5件を返却する

具体的な実装は、以下の通りです。


```python
# 指定したユーザーへレコメンドするアイテムを5個出力する関数
def get_recommend_items(user_id):
    # 高く評価した映画のリストを取得
    favorite = df.loc[:, user_id].nonzero()
    # 評価テーブルから、高評価の行を取り出す
    table = d[favorite]
    # 列ごとに類似度を合計
    table[np.where(table < 0)] = 0
    indicator = table.sum(axis=0)
    # 類似度の高い順にソート
    sorted_id = indicator.argsort()[::-1]
    # 評価済み映画のリストを取得
    reviewed = raw[raw.loc[:, user_id].notnull()].index.tolist()
    # 評価済みを削除
    recommend_id = [i for i in sorted_id if i not in reviewed]
    # 5件だけ返す
    return recommend_id[:5]

# 試しにUser_ID=100の人
recommends = get_recommend_items(100)
print(recommends)
```

    [312, 301, 299, 306, 257]


これで、レコメンド処理の実装が完了しました！！！

## 5. レコメンド結果の評価
上記で作成したレコメンドモデルについて、どれほど良いのか（悪いのか）評価したいと思います。  
ここでは評価用のデータ（u1.test）を用いて評価を行います。

**[評価方法]**
* 評価データに存在するユーザーに対して、レコメンドを5件表示する.
  * レコメンドの生成は、上記で定義した「**get_recommend_items**」を用います.
* 表示したレコメンド5件のうち、1件以上、評価データ内で閲覧したデータがあれば成功とする.
* 「成功数 / ユーザー数」で精度を測る.

まずはテストデータを読み込みます。


```python
utest = pd.read_csv("data/ml-100k/u1.test", delimiter="\t", names=("user", "movie", "rating", "timestamp"))
utest.head()
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
      <th>0</th>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>887431973</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>875693118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>12</td>
      <td>5</td>
      <td>878542960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14</td>
      <td>5</td>
      <td>874965706</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>17</td>
      <td>3</td>
      <td>875073198</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 好評価のみを対象とした、行列（行=映画、列=ユーザー）を作成します.
high_rate_test = utest.loc[udata["rating"] >= 3]
raw_test = high_rate_test.pivot(index="movie", columns="user", values="rating")
df_test = raw_test.fillna(0)
df_test = df_test.where(df_test < 3, 1)
```


```python
### 試しに、userId=1の人でテスト.
user_id = 1
# (1) レコメンド対象
recommends = set(get_recommend_items(user_id))
# (2) テストデータ内に存在する閲覧データ
real = set(df_test.loc[:, user_id].nonzero()[0])
# (1) と (2) の積集合
real & recommends
```




    {97}



無事にレコメンドができたようです（ホッとしますw）。

続けて、他の人も評価を行なっていきましょう。


```python
# テストデータに存在するユーザーの一覧を取得する.
users = df_test.columns

# 全ユーザー数
all = len(users)

# 成功数
good = 0

# 1ユーザーごとに、成功 or not を判定する.
for user_id in users:
    real = set(df_test.loc[:, user_id].nonzero()[0])
    recommends = set(get_recommend_items(user_id))
    matches = real & recommends
    good += 1 if matches else 0

# 結果を表示.
print("全件={0}, 成功数={1}, 成功率={2}%".format(all, good, good * 100 // all))
```

    全件=458, 成功数=239, 成功率=52%


今回の場合には、52%の確率で、ユーザーが将来閲覧する映画をレコメンドすることができました。

めでたしめでたし。


```python

```
