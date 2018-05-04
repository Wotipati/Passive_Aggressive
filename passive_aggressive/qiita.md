## Abstract
オンライン機械学習のアルゴリズムの一つである**Passive Aggressive (PA)**をpythonで実装しました。  
学習の過程を可視化することにより、オンライン機械学習の欠点の一つである**ノイズに弱い**という点を実感し、その解決案を考えてみます。  

![ezgif com-resize](https://user-images.githubusercontent.com/26996041/39611189-d019a09c-4f90-11e8-853c-75d6ef24bc02.gif)


あんまり理論についての詳しい解説ではないです。
詳しい理論をしっかり学びたい方は[オンライン機械学習 (機械学習プロフェッショナルシリーズ)](https://www.kspub.co.jp/book/detail/1529038.html)に詳しく書いてあるので、そちらを参考にしていただければ幸いです。

この動画を見て、実装してみたいなーと思った人が対象かなと思います。
このページと[結果を出力するコードが全部乗っているgithubリポジトリ](https://github.com/Wotipati/Online_Machine_Learning/tree/master/passive_aggressive)が誰かの参考になれば幸いです。



## オンライン機械学習とは？
一言で表すと
**データが与えられる度に逐次的に学習を行う手法**
です。
これに対し、既に存在している**データ全体を利用してまとめて学習を行う手法**をバッチ学習といいます。

## Passive Aggressive (PA)
PAはオンライン学習を用いた線形分類器の一種で2006年に発表されました。　　
今回はPAを利用した単純な二値分類の学習問題を扱います。　　

二値分類ではデータが入力$\mathbf{x}$から二値の出力$y \in$ {-1, +1}を推定します。
その中でも線形分類器はモデルパラメタ$\mathbf{W}$で特徴づけられ、入力$\mathbf{x}$に対し、

```math
sign(\mathbf{W}^{\mathrm{T}}\mathbf{x})
```
を推定します。
$sign(x)$は$x$が0以上ならば１を、負ならば-1を返す関数です。

### 更新式
オンライン学習ではデータが与えられるたびに、事前に設計した更新式を用いて、パラメタを更新していきます。
PAでは$t$番目のデータ$(\mathbf{x}^{(t)}, y^{(t)})$が与えられた時、パラメタ$\mathbf{W}^{(t)}$を、以下の式を用いて$\mathbf{W}^{(t+1)}$に更新します。  


```math
\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} +\frac{l_{hinge}(\mathbf{x}^{(t)}, y^{(t)}, \mathbf{W}^{(t)})}{\|\mathbf{x}^{(t)}\|^2}y^{(t)}\mathbf{x}^{(t)}
```
```math
l_{hinge}(\mathbf{x}, y, \mathbf{W}) = max(0, 1 - y\mathbf{W}^{\mathrm{T}}\mathbf{x})
```
詳しい導出は[オンライン機械学習 (機械学習プロフェッショナルシリーズ)](https://www.kspub.co.jp/book/detail/1529038.html)を参照していただければと思いますが、
とりあえず、

新しく入ってきたデータに対し、  
<現在のモデルで十分なマージンで分類できる（$y\mathbf{W}^{\mathrm{T}}\mathbf{x}>1$）場合>
$l_{hinge}(\mathbf{x}, y, \mathbf{W}) = 0$となり、**更新が行われない**   
<それ以外の場合>
**間違えた割合に応じて更新幅を変えて更新** される   
ということを意識においていただければと思います。  
例えば判定を間違えた場合は $y\mathbf{W}^{\mathrm{T}}\mathbf{x}<0$ からの、 $l_{hinge}(\mathbf{x}, y, \mathbf{W}) > 1$ となり、パラメタが大きく更新されます。


### 実装
`PassiveAggressive`というクラスを作って実装しました。

```python
class PassiveAggressive:
    def __init__(self):
        self.t = 0
        self.w = None

    def L_hinge(self, vec_x, y):
        return max([0, 1-y*self.w.dot(vec_x)])

    def calc_eta(self, loss, vec_x):
        l2_norm = vec_x.dot(vec_x)
        return loss/l2_norm

    def update(self, vec_x, y):
        loss = self.L_hinge(vec_x, y)
        eta = self.calc_eta(loss, vec_x)
        self.w += eta*y*vec_x
        self.t += 1

    def fit(self, vec_feature, y):
        weight_dim = len(vec_feature)
        if self.w is None:
            self.w = np.ones(weight_dim)
        self.update(vec_feature, y)
```
後ほどこのクラスを利用してPA-1というアルゴリズムを実装するため、`calc_eta`という関数にわけて実装してあります。  

使うときは

```python
# データセットの用意
train_dataset = SimpleDataset(total_num=1000, is_confused=True, x=3, y=5, seed=1)

# モデルの用意
model = PassiveAggressive()

# データが入力されるたびにモデルを更新
for i in range(len(train_dataset.y)):
        model.fit(train_dataset.feature_vec[i], train_dataset.y[i])
```
というような流れでオンライン学習を行います。


### PAの問題点
実はこのアルゴリズムは**新しく入ってきたデータはかならず正しく分類されるようにパラメタを更新する**という非常に強い制約が加えられてます。  


これにより、**ノイズのようなデータ** が入ってきた時、
これまで頑張って学習してきた結果を全て捨てて、ノイズデータにあわせてパラメタを大きく更新してしまう。
という事態が起こってしまいます。  

これはPAに限らず、オンライン学習全体の短所と言える部分で、一般にオンライン学習はバッチ学習よりもノイズに弱いと言われています。

冒頭の動画では、左の図が推定した決定境界と使用した訓練データをプロットしたもの、右の図がその決定境界をテストデータを用いて評価した結果を表していますが、
青色のPAに注目すると、ノイズが入ってきたとき、境界の直線も大きく移動し、評価結果も大きく下がっていることがわかるかと思います。


## PA-1: PAの改良版
PAが発表された論文では、このノイズに弱いという点を改善するアルゴリズムとしてPA-1が提案されています。  

PA-1の更新式は以下のようになります。

```math
\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} +min\left(C, \frac{l_{hinge}(\mathbf{x}^{(t)}, y^{(t)}, \mathbf{W}^{(t)})}{\|\mathbf{x}^{(t)}\|^2}\right)y^{(t)}\mathbf{x}^{(t)}
```

更新幅が$C$を超える場合は$C$にクリップするというものです。
単純です。

### PA-1の実装
先ほど実装した`PassiveAggressive`を継承し、`PassiveAggressiveOne`を作ります。

```python
class PassiveAggressiveOne(PassiveAggressive):
    def __init__(self, c=0.1):
        self.c = c
        PassiveAggressive.__init__(self)

    def calc_eta(self, loss, vec_x):
        l2_norm = vec_x.dot(vec_x)
        return min(self.c, loss/l2_norm)
```

これも簡単です。


## 結果
では改めて結果を見てみましょう。  

![ezgif com-resize](https://user-images.githubusercontent.com/26996041/39611189-d019a09c-4f90-11e8-853c-75d6ef24bc02.gif)

青が改良前のPA、赤が改良後のPA-1です。  
PA-1の方がノイズに対しても安定していることがひと目でわかると思います。　　

最後までお読み頂き、ありがとうございます。  


## Appendix
可視化に使用したコードなどは[github](https://github.com/Wotipati/Online_Machine_Learning/tree/master/passive_aggressive)にあげてあります。
(今後も随時、オンライン機械学習のアルゴリズムを実装してアップロードしていく予定です)

使い方はcloneしたあとに

```
$ python passive_aggressive_ex.py
```
としてくれれば、結果が表示されます。
このpassive_aggressive_ex.pyにPA-1が実装されてます。  


また、結果を動画に保存したい場合は

```
$ python passive_aggresive_ex.py --record 1
```
とすれば"results.mp4"という名前で保存されます。  

動画保存の際に[AtsushiSakai/matplotrecorder](https://github.com/AtsushiSakai/matplotrecorder)というスクリプトを使用させて頂きました。


また、実験で使用するデータセットは[統計の素人だけどPythonで機械学習モデルを実装したい、そんな人のための第一歩](https://qiita.com/hik0107/items/9b6e1e989f4eaefdc31d)を参考にsimple_dataset.pyというスクリプトに実装しました。


## References
[オンライン機械学習 (機械学習プロフェッショナルシリーズ)](https://www.kspub.co.jp/book/detail/1529038.html)
基本的な理論についてはこの本を参考にしてあります。
今回はかなりざっくりとした解説しかしないので、詳しい理論をしっかり学びたい方はこの本を読んでいただければと思います。  

[統計の素人だけどPythonで機械学習モデルを実装したい、そんな人のための第一歩](https://qiita.com/hik0107/items/9b6e1e989f4eaefdc31d)
データを用意する際に参考にさせて頂きました。

[実装して理解するオンライン学習器(1) - PassiveAggresive](http://smrmkt.hatenablog.jp/entry/2014/10/13/124757)
アルゴリズム自体の実装をする際に参考にさせて頂きました。
