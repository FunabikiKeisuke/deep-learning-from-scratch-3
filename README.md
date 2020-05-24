## リポジトリについて
 このリポジトリは[『ゼロから作る Deep Learning ❸』内に書かれているソースコード](https://github.com/oreilly-japan/deep-learning-from-scratch-3) と、一部私が追加したソースコードから成っています。
 
 著作権は本書の著者に帰属します。
 
 <p>
  <a href="https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/LICENSE.md"><img
		alt="MIT License"
		src="http://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>


<p align="center">
<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/deep-learning-from-scratch-3.png" width="200px">
</p>


## フレームワーク編

本書では「DeZero」というディープラーニングのフレームワークを作ります。DeZeroは本書オリジナルのフレームワークです。最小限のコードで、フレームワークのモダンな機能を実現します。本書では、この小さな——それでいて十分にパワフルな——フレームワークを、全部で60のステップで完成させます。それによって、PyTorch、TensorFlow、Chainerなどの現代のフレームワークに通じる深い知識を養います。

<p>
<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/dezero_logo.png" width="400px">
</p>


## ファイル構成

|フォルダ名 |説明         |
|:--        |:--                  |
|[dezero](/dezero)       |DeZeroのソースコード|
|[examples](/examples)     |DeZeroを使った実装例|
|[steps](/steps)|各stepファイル（step01.py ~ step60.py）|
|[tests](/tests)|DeZeroのユニットテスト|


## 必要な外部ライブラリ

本書で使用するPytnonのバージョンと外部ライブラリは下記の通りです。

- [Python 3系](https://docs.python.org/3/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

またオプションとして、NVIDIAのGPUで実行できる機能も提供します。その場合は下記のライブラリが必要です。

- [CuPy](https://cupy.chainer.org/) （オプション）


## 実行方法

本書で説明するPythonファイルは、主に[steps](/steps)ファルダにあります。
実行するためには、下記のとおりPythonコマンドを実行します（どのディレクトリからでも実行できます）。

```
$ python steps/step01.py
$ python steps/step02.py

$ cd steps
$ python step31.py
```
