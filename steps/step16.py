import numpy as np


class Variable:
    """「箱」に変数を入れる。

    Attributes:
        data (numpy.ndarray): 格納する変数。
        grad (NoneType or numpy.float64): 逆伝播された微分値。
        creator (NoneType or Function): 変数を生み出した関数を記憶している変数。
        generation (Int): 変数の世代を記憶している変数。
    """

    def __init__(self, data):
        """
        Args:
            data (numpy.ndarray): 格納する変数。

        Raises:
            TypeError: numpy.ndarray以外の型を引数として受け取った場合。
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        """変数を生み出した関数とその世代をセットする。"""
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        """設定した微分値をリセットする。"""
        self.grad = None

    def backward(self):
        """合成関数の逆伝播をループで処理する。"""
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            """逆伝播をする関数の順番を世代で並び替える。"""
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 1. 変数を生み出した関数を取得する。
            gys = [output.grad for output in f.outputs]  # 2. 変数を生み出した関数の出力値を取得する。
            gxs = f.backward(*gys)  # 3. 変数を生み出した関数の逆伝播を呼び出す。
            if not isinstance(gxs, tuple):
                gxs = gxs,

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # 既に微分値がセットされていたら和を取る。

                if x.creator is not None:
                    add_func(x.creator)


def as_array(x):
    """numpy.ndarray以外の型をnumpy.ndarrayに変換する。"""

    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """値を受け取って順伝播と逆伝播を計算する。

    Attributes:
        inputs (tuple): 関数へ入力する値。
        outputs (list): 関数から出力する値。
        generation (Int): 関数の世代。

    Notes:
        継承する必要あり。
    """

    def __call__(self, *inputs):
        """
        Args:
            *inputs (Variable): 関数へ入力する値が入っているインスタンス。

        Returns:
            outputs (Variable): 関数の処理結果を入れたインスタンス。
        """
        xs = [x.data for x in inputs]  # Variableからdataを取得する。
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):  # forwardの返り値がtuple以外ならtupleにする。
            ys = ys,
        outputs = [Variable(as_array(y)) for y in ys]  # dataをlistで包む。

        self.generation = max([x.generation for x in inputs])  # 変数の最大の世代を関数の世代とする。
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    """x ** 2の順伝播と逆伝播をする。"""

    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Add(Function):
    """x0 + x1 の順伝播と逆伝播をする。"""

    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
