import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    # 生成空白图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 定义生成值的个数
    n = 250

    # 生成 lambda函数来生成给定范围的值
    f = lambda min_val, max_val, n: min_val + (max_val - min_val) * np.random.randn(n)

    # 生成值
    x_val = f(15, 41, n)
    y_val = f(-10, 70, n)
    z_val = f(-52, -37, n)

    # 绘制图像
    ax.scatter(x_val, y_val, z_val, c='k', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()