from torch import Tensor
import numpy as np
from matplotlib import pyplot as plt
from zorro import Zorro

if __name__ == '__main__':
    zorro = Zorro()

    x = np.arange(-5, 5, 0.1)
    y = zorro.forward(Tensor(x)).numpy()

    plt.title("Función Zorro")
    plt.xlabel("x")
    plt.ylabel("z(x)")
    plt.plot(x, y)
    plt.show()
