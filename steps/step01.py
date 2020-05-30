import numpy as np


class Variable:
    """「箱」に変数を入れる。"""

    def __init__(self, data):
        """
        Args:
            data (ndarray): 入力変数
        """
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
