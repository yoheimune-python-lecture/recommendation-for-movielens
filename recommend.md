
dataフォルダがない場合に作成する処理を追加


```python
import os
from urllib.request import urlopen

# MovieLensのサイトから、Zipファイルを取得し、ローカルに保存します.

# dataフォルダがない場合に作成する処理を追加
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



pivotメソッドを使用


```python
high_rate = udata.loc[udata["rating"] >= 3]
# movieを行, columnsを列にした後、欠損部分を0埋め
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




```python
# 上記の雰囲気で、総当たりで全アイテムの距離を計算する.
from scipy.spatial.distance import pdist
d = pdist(df, "cosine")
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


例えば、映画ID=1に類似する映画を、類似度の高い順に並べてみます。

メモリ効率や速度を考え、Numpyを使います。


```python
movie_id = 0

# 評価の良い順に並べます.
# ソート後のインデックスを収納
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


レコメンドの関数


```python
# 指定したユーザーへレコメンドするアイテムを5個出力する関数
def get_recommend_items(user_id):
    # 高く評価した映画のリストを取得
    favorite = df.loc[:, user_id].nonzero()
    print(len(favorite[0]))
    # 評価済み映画のリストを取得
    reviewed = raw[raw.loc[:, user_id].notnull()].index.tolist()
    # 仮
    reviewed = [r - 1 for r in reviewed]
#     print(reviewed)
#     print("---------")
    # 評価テーブルから、高評価の行を取り出す
    table = d[favorite]
#     print(table.shape)
#     print("-------------------")
#     print(table < 0)
#     print("-------------------")
#     print(table)
    # 列ごとに類似度を合計
    table[np.where(table < 0)] = 0
    indicator = table.sum(axis=0)
#     print("-------------------")
#     print(indicator.shape)
#     print(indicator[indicator.argsort()[::-1]][:7])
#     print(indicator.argsort())
    # ソート
    sorted_id = indicator.argsort()[::-1]
    # 評価済みを削除
    recommend_id = [i for i in sorted_id if i not in reviewed]
    #print(recommend_id)
    # 5件だけ返す
    return recommend_id[:5]

# 試しにUser_ID=100の人
recommends = get_recommend_items(100)
print(recommends)
```

    25
    [312, 301, 306, 313, 331]


テスト


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
high_rate_test = utest.loc[udata["rating"] >= 3]
raw_test = high_rate_test.pivot(index="movie", columns="user", values="rating")
df_test = raw_test.fillna(0)
df_test = df_test.where(df_test < 3, 1)
```


```python
# 試しに、userId=1の人でテスト.
user_id = 1
real = set(df_test.loc[:, user_id].nonzero()[0])
recommends = set(get_recommend_items(user_id))
real & recommends
```

    108





    {95, 97, 173}




```python
users = df_test.columns
all = len(users)
good = 0

for user_id in users:
    real = set(df_test.loc[:, user_id].nonzero()[0])
    recommends = set(get_recommend_items(user_id))
    matches = real & recommends
    good += 1 if matches else 0

print("全件={0}, 成功数={1}, 成功率={2}%".format(all, good, good * 100 // all))
```

    108
    37
    16
    13
    59
    91
    193
    24
    11
    94
    96
    25
    250
    38
    41
    65
    14
    155
    9
    20
    45
    52
    79
    41
    41
    46
    9
    34
    16
    23
    22
    19
    14
    7
    12
    8
    30
    48
    11
    14
    29
    76
    100
    68
    25
    11
    14
    29
    56
    9
    14
    35
    15
    23
    9
    82
    48
    65
    193
    119
    7
    94
    35
    96
    47
    20
    11
    11
    34
    70
    20
    65
    26
    20
    30
    34
    28
    11
    25
    13
    28
    63
    60
    40
    144
    9
    106
    11
    38
    153
    42
    160
    6
    189
    127
    26
    30
    13
    53
    25
    28
    58
    13
    36
    11
    27
    8
    14
    106
    64
    9
    24
    29
    25
    46
    59
    33
    40
    89
    11
    36
    31
    26
    13
    73
    19
    9
    89
    8
    166
    14
    17
    14
    10
    18
    20
    21
    27
    14
    9
    50
    17
    13
    93
    132
    16
    10
    29
    17
    9
    153
    65
    7
    25
    5
    17
    23
    87
    51
    63
    15
    18
    10
    30
    18
    8
    32
    25
    23
    15
    16
    7
    26
    77
    20
    36
    59
    136
    14
    26
    26
    14
    22
    123
    23
    36
    33
    55
    98
    29
    10
    15
    49
    111
    37
    17
    38
    87
    14
    106
    162
    9
    17
    16
    6
    9
    111
    10
    10
    68
    17
    66
    59
    48
    67
    24
    30
    14
    10
    75
    143
    39
    51
    15
    28
    25
    4
    10
    67
    10
    45
    66
    207
    42
    55
    27
    12
    74
    11
    9
    11
    43
    101
    11
    73
    14
    14
    87
    55
    42
    11
    54
    65
    19
    110
    29
    6
    27
    12
    9
    67
    55
    57
    21
    12
    101
    119
    93
    67
    137
    30
    11
    36
    41
    238
    23
    12
    184
    127
    14
    9
    27
    26
    15
    139
    25
    34
    7
    69
    146
    55
    153
    75
    98
    68
    79
    77
    135
    10
    133
    5
    206
    12
    103
    12
    49
    211
    10
    2
    137
    128
    137
    121
    42
    29
    10
    84
    15
    74
    72
    27
    40
    51
    65
    79
    135
    148
    31
    84
    34
    107
    13
    193
    10
    59
    19
    38
    150
    23
    10
    105
    141
    94
    125
    104
    94
    30
    22
    34
    29
    25
    13
    143
    20
    14
    49
    29
    14
    66
    69
    10
    136
    11
    28
    16
    33
    25
    16
    51
    37
    43
    143
    142
    15
    22
    19
    230
    126
    87
    75
    27
    51
    16
    151
    18
    157
    40
    175
    15
    82
    74
    281
    112
    40
    32
    66
    115
    189
    15
    81
    49
    37
    29
    141
    227
    164
    20
    129
    17
    48
    35
    34
    20
    20
    362
    238
    7
    28
    32
    47
    69
    45
    28
    120
    83
    23
    46
    307
    51
    16
    42
    28
    33
    270
    117
    197
    26
    26
    43
    96
    18
    30
    20
    109
    24
    56
    467
    50
    148
    113
    149
    171
    190
    244
    150
    96
    56
    27
    全件=458, 成功数=267, 成功率=58%



```python

```
