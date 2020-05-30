import numpy as np


class Variable:
    """「箱」に変数を入れる。"""

    def __init__(self, data):
        """
        Args:
            data (ndarray): 格納する変数。
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
            in_data (Variable.data): 関数へ入力する値。

        Raises:
            NotImplementedError: classを継承せずに呼び出した場合。
        """
        raise NotImplementedError()


class Square(Function):
    """受け取った値を二乗して返す関数。"""

    def forward(self, x):
        """
        Args:
            x (ndarray): 関数へ入力する値。

        Returns:
            x ** 2 (ndarray): 入力値を二乗した値。

        """
        return x ** 2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
