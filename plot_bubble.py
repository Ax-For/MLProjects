import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    num_val = 40

    x = np.random.rand(num_val)
    y = np.random.rand(num_val)

    max_radius = 25
    area = np.pi * (max_radius * np.random.rand(num_val)) ** 2

    colors = np.random.rand(num_val)
    plt.scatter(x, y, s=area, c=colors, alpha=1.0)

    plt.show()