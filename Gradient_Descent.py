# Gradient Descent,梯度下降法,求解最优值.
# 目标函数用最小二乘法.
# 递归更新参数,求解最优值.
# 偏导方向是最速下降方向.

# Batch Gradient Descent 批梯度下降
# 特点：速度慢,每次迭代便利整个集合.
# 关于参数迭代：每次使用上一次所有的参数更新下一代的参数
# 关于学习速率：最好先把特征规范化到指定区间,再选取学习速率,能收敛,并且速率要恰当,
import numpy as np
import matplotlib.pyplot as plt

def Batch_Gradient_Descent():
    area = [2104,1600,2400,1416,3000] # x
    price = [400,330,369,232,540]     # y
    theta = [0,0]
    alpha = 0.00000001                # 学习速率

    iterations = 10                          # 迭代20次

    x = np.linspace(1400, 3000, 100)
    h = theta[0] + theta[1] * x
    plt.plot(area, price, 'o')
    plt.plot(x, h)
    plt.xlabel('Area(feet^2)')
    plt.ylabel('Price(1000$s)')
    plt.title("Btach Gradient Descent")

    while iterations>0:
        iterations = iterations - 1;
        partial0 = 0  # 迭代中的更新参数
        partial1 = 0
        for j in range(0,len(area)):
            partial0 = partial0 + (theta[0] + theta[1] * area[j] - price[j]);
            partial1 = partial1 + (theta[0] + theta[1] * area[j] - price[j]) * area[j];
        theta[0] = theta[0] - alpha * partial0
        theta[1] = theta[1] - alpha * partial1

        h = theta[0] + theta[1] * x
        plt.plot(x, h)

    plt.savefig('Batch Gradient Descent.png', dpi=1000)
    plt.show()
    plt.close()

# Stochastic Gradient Descent 随机梯度下降
# 特点：收敛速度快,每次用一个数据更新所有参数,每个数据只用一次.
# 由于数据量较小,再此以所有数据迭代一遍为基本单位绘图,
# 当数据量特别大时,例如十万条,批梯度下降每次迭代需要遍历十万条数据,
# 而随机梯度下降每次只利用一个数据,直到收敛为止,所以速度快.
def Stochastric_Gradient_Descent():
    area = [2104, 1600, 2400, 1416, 3000]  # x
    price = [400, 330, 369, 232, 540]  # y
    theta = [0, 0]
    alpha = 0.00000001                # 学习速率

    x = np.linspace(1400, 3000, 100)
    h = theta[0] + theta[1] * x
    plt.plot(area, price, 'o')
    plt.plot(x, h)
    plt.xlabel('Area(feet^2)')
    plt.ylabel('Price(1000$s)')
    plt.title("Stochastric Gradient Descent")

    iterations = 10                           # 预设迭代十次停止.
    while iterations>0:
        iterations = iterations - 1
        for i in range(0,len(area)):        # 每次利用一个数据更新参数.
            partital0 = alpha * (theta[0] + theta[1] * area[i] - price[i])
            partital1 = alpha * (theta[0] + theta[1] * area[i] - price[i]) * area[i]

            theta[0] = theta[0] - partital0
            theta[1] = theta[1] - partital1

        h = theta[0] + theta[1] * x
        plt.plot(x, h)

    plt.savefig('Stochastric Gradient Descent.png', dpi=1000)
    plt.show()
    plt.close()

Batch_Gradient_Descent()
Stochastric_Gradient_Descent()
