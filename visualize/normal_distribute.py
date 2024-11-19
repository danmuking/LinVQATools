import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义正态分布的均值和标准差
mu = 0  # 均值
sigma = 1  # 标准差

# 生成 x 值范围
x = np.linspace(-5, 5, 1000)

# 计算正态分布的概率密度函数值
y = norm.pdf(x, mu, sigma)

# 绘制正态分布图
plt.plot(x, y, color='green', label=f'N({mu},{sigma}^2)',linewidth=5)
# plt.xlabel('x')
# plt.ylabel('Probability Density')
# plt.title('Normal Distribution')
# plt.legend()
# plt.grid()
plt.xticks([])
plt.yticks([])
# 仅保留x轴和y轴的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-0.1, 0.5)
# 添加箭头效果的 x 和 y 轴
arrowprops = dict(arrowstyle='->', linewidth=5, color='black')
ax.annotate('', xy=(1.05, 0), xycoords='axes fraction', xytext=(0, 0),
            textcoords='axes fraction', arrowprops=arrowprops)
ax.annotate('', xy=(0, 1.05), xycoords='axes fraction', xytext=(0, 0),
            textcoords='axes fraction', arrowprops=arrowprops)
# 显示图形
plt.show()
