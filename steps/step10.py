import unittest
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
        """
        Args:
            func (Function): 変数を生み出した関数。
        """
        self.creator = func

    def backward(self):
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

    def __call__(self, input):
        """
        Args:
            input (Variable): 関数へ入力する値が入っているインスタンス。

        Returns:
            output (Variable): 関数の処理結果を入れたインスタンス。
        """
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # 変数に生みの親（関数）をセットする。
        self.input = input
        self.output = output  # 生み出した変数をセットする。
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    """x ** 2の順伝播と逆伝播をする。"""

    def forward(self, x):
        """
        Args:
            x (numpy.ndarray or numpy.float64): 関数へ入力する値。

        Returns:
            y (numpy.float64): 入力値を二乗した値。
        """
        y = x ** 2
        return y

    def backward(self, gy):
        """
        Args:
            gy (numpy.ndarray or numpy.float64): 逆伝播してきた値。

        Returns:
            gx (numpy.float64): 逆伝播する値。
        """
        x = self.input.data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


def numerical_diff(f, x, eps=1e-4):
    """
    Args:
        f (Function): 数値微分する関数。
        x (Variable): 数値微分する値。
        eps (float, default 1e-4): 微小な値。

    Returns:
        (numpy.float64): 数値微分の結果。
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    """class Squareのテスト。"""

    def test_forward(self):
        x = Variable(np.array(4.0))
        y = square(x)
        expected = np.array(16.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(7.0))
        y = square(x)
        y.backward()
        expected = np.array(14.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


# step10.py
unittest.main()
