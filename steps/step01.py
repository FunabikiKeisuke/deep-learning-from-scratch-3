import numpy as np


class Variable:
    """
    Put variables in the box
    """

    def __init__(self, data):
        """
        Args:
            data (ndarray): Input variable
        """
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
