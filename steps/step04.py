import numpy as np


class Variable:
    """「箱」に変数を入れる。"""

    def __init__(self, data):
        """
        Args:
            data (numpy.float64): 格納する変数。
        """
        self.data = data


class Function:
    """値を受け取って処理をする関数。

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
        output = Variable(y)
        return output

    def forward(self, in_data):
        """
        Args:
            in_data (numpy.float64): 関数へ入力する値。

        Raises:
            NotImplementedError: classを継承せずに呼び出した場合。
        """
        raise NotImplementedError()


class Square(Function):
    """受け取った値を二乗して返す関数。"""

    def forward(self, x):
        """
        Args:
            x (numpy.float64): 関数へ入力する値。

        Returns:
            x ** 2 (numpy.float64): 入力値を二乗した値。

        """
        return x ** 2


class Exp(Function):
    """受け取った値のexponetialを取って返す関数。"""

    def forward(self, x):
        """
        Args:
            x (numpy.float64): 関数へ入力する値。

        Returns:
            np.exp(x) (numpy.float64): 入力値をexponetialで取った値。
        """
        return np.exp(x)


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


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


def f(x):
    """
    Args:
        x (Variable): 合成関数のx。

    Returns:
        (Variable): 合成関数を返す。
    """
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
