import numpy as np


class Variable:
    """「箱」に変数を入れる。

    Attributes:
        data (numpy.ndarray): 格納する変数。
        grad (NoneType or numpy.float64): 逆伝播された微分値。
        creator (NoneType or Function): 変数を生み出した関数を記憶する変数。
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

    def set_creator(self, func):
        """変数を生み出した関数をセットする。"""
        self.creator = func

    def backward(self):
        """合成関数の逆伝播をループで処理する。"""
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 1. 変数を生み出した関数を取得する。
            x, y = f.input, f.output  # 2. 変数を生み出した関数の入力値と出力値を取得する。
            x.grad = f.backward(y.grad)  # 3. 変数を生み出した関数の逆伝播を呼び出す。

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    """numpy.ndarray以外の型をnumpy.ndarrayに変換する。"""

    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """値を受け取って順伝播と逆伝播を計算する。

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

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    """x0 + x1 の順伝播をする。"""

    def forward(self, x0, x1):
        y = x0 + x1
        return y


def add(x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)
