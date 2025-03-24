#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def logistic_regression_visualization(x, y):
    """
    带美观可视化的逻辑回归预测函数
    参数：
    x : 一维特征列表
    y : 一维目标值列表（连续值范围建议在0-1之间）
    """
    # 转换为numpy数组并重塑维度
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    # 创建带数据标准化的SVM回归管道
    model = make_pipeline(
        StandardScaler(),
        SVR(kernel='poly', C=100, epsilon=0.1)
    )

    # 训练模型
    model.fit(x, y)

    # 生成预测数据
    x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # 创建渐变颜色映射（浅红到深红）
    red_gradient = LinearSegmentedColormap.from_list(
        'red_gradient', ['#ffcccc', '#cc0000'], N=256
    )

    # 设置专业级可视化参数
    # 设置无网格的干净样式
    sns.set(style="white", rc={
        'figure.figsize': (10, 6),
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False  # 明确关闭网格
    })

    fig, ax = plt.subplots(dpi=120)

    # 绘制原始数据点
    ax.scatter(x, y, c='#2c7bb6', edgecolor='white',
               s=60, alpha=0.8, label='观测数据')

    # 创建渐变曲线
    points = np.array([x_range.ravel(), y_pred]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=red_gradient,
                        linewidth=3, alpha=0.9)
    lc.set_array(x_range.ravel())  # 根据x值设置颜色渐变

    # 添加渐变曲线
    plt.gca().add_collection(lc)

    # 添加颜色条（可选）
    # cbar = plt.colorbar(lc, ax=ax)
    # cbar.set_label('特征X值', rotation=270, labelpad=15)

    # 设置坐标轴和标签
    ax.set(xlim=(x.min()-0.1, x.max()+0.1),
           ylim=(y.min()-0.1, y.max()+0.1),
           xlabel='Predicted Score',
           ylabel='Mos',)


    # 添加图例
    # ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.show()

# 示例数据生成（S型曲线数据）
if __name__ == "__main__":
    true_list = []
pred_list = []

# 打开并读取文件
with open('../score.txt', 'r') as file:
    for line in file:
        # 分割每行内容
        parts = line.strip().split()
        # 提取真实值和预测值，并转换为浮点数
        true_value = float(parts[1])
        pred_value = float(parts[2])
        # 添加到对应的列表中
        true_list.append(true_value)
        pred_list.append(pred_value)

    # 转换为NumPy数组
    true_array = np.array(true_list)
    pred_array = np.array(pred_list)

    logistic_regression_visualization(pred_array, true_array)