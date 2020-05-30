import numpy as np


class Variable:
    """「箱」に変数を入れる。

    Attributes:
        data (numpy.ndarray or numpy.float64): 格納する変数。
        grad (NoneType or numpy.float64): 逆伝播された微分値。
        creator (NoneType or Function): 変数を生み出した関数を記憶する変数。
    """

    def __init__(self, data):
        """
        Args:
            data (numpy.ndarray or numpy.float64): 格納する変数。
        """
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
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 1. 変数を生み出した関数を取得する。
            x, y = f.input, f.output  # 2. 変数を生み出した関数の入力値と出力値を取得する。
            x.grad = f.backward(y.grad)  # 3. 変数を生み出した関数の逆伝播を呼び出す。

            if x.creator is not None:
                funcs.append(x.creator)


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
        output.set_creator(self)  # 変数に生みの親（関数）をセットする。
        self.input = input
        self.output = output  # 生み出した変数をセットする。
        return output

    def forward(self, x):
        """
        Args:
            x (numpy.ndarray or numpy.float64): 関数へ入力する値。

        Raises:
            NotImplementedError: classを継承せずに呼び出した場合。
        """
        raise NotImplementedError()

    def backward(self, gy):
        """
        Args:
            gy (numpy.ndarray or numpy.float64): 逆伝播してきた値。

        Raises:
            NotImplementedError: classを継承せずに呼び出した場合。
        """
        raise NotImplementedError()


class Square(Function):
    """受け取った値を二乗して返す関数。"""

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


class Exp(Function):
    """受け取った値のexponetialを取って返す関数。"""

    def forward(self, x):
        """
        Args:
            x (numpy.float64): 関数へ入力する値。

        Returns:
            y (numpy.float64): 入力値をexponetialで取った値。
        """
        y = np.exp(x)
        return y

    def backward(self, gy):
        """
        Args:
            gy (numpy.float64): 逆伝播してきた値。

        Returns:
            gx (numpy.float64): 逆伝播する値。
        """
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# backward
y.grad = np.array(1.0)
y.backward()
print(x.grad)
