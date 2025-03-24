import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def plot_chunk(ax, x_chunk, y_chunk, chunk_id,title):
    """
    在指定axes上绘制单个分块图表
    :param ax: matplotlib的axes对象
    :param x_chunk: 特征数据
    :param y_chunk: 目标数据
    :param chunk_id: 分块编号
    """
    # 数据预处理
    x = np.array(x_chunk).reshape(-1, 1)
    y = np.array(y_chunk)

    # 训练SVR模型
    model = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf',         # 改用径向基核函数
            C=1.0,                # 调整正则化强度
            epsilon=0.2,          # 扩展容忍区域
            gamma='scale')        # 自动计算核系数
    )
    model.fit(x, y)

    # 生成预测数据
    x_range = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # 创建渐变颜色映射
    red_gradient = LinearSegmentedColormap.from_list(
        'red_gradient', ['#ffcccc', '#cc0000'], N=256
    )

    # 绘制数据点
    ax.scatter(x, y, c='#2c7bb6', edgecolor='white',
               s=15, alpha=0.8, linewidth=0.5)
    ax.set_title(title, fontsize=6, pad=2, loc='left')


    # 创建渐变曲线
    points = np.array([x_range.ravel(), y_pred]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=red_gradient, linewidth=1.5, alpha=0.9)
    lc.set_array(x_range.ravel())
    ax.add_collection(lc)

    # 设置坐标轴范围
    ax.set_xlim(x.min()-0.1, x.max()+0.1)
    ax.set_ylim(y.min()-0.1, y.max()+0.1)

    # # 添加分块编号
    # ax.text(0.05, 0.95, f'#{chunk_id+1}', transform=ax.transAxes,
    #         fontsize=6, ha='left', va='top',
    #         bbox=dict(facecolor='white', alpha=0.8, pad=1))
    #
    # # 添加R²分数
    # r2 = model.score(x, y)
    # ax.text(0.95, 0.05, f'R²={r2:.2f}', transform=ax.transAxes,
    #         fontsize=6, ha='right', va='bottom',
    #         bbox=dict(facecolor='white', alpha=0.8, pad=1))

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=5, pad=1)
    ax.grid(False)

def create_mosaic(data_length=540, chunk_size=9):
    """
    创建拼接大图
    :param data_length: 总数据量（默认540）
    :param chunk_size: 分块大小（默认9）
    """
    # 生成示例数据
    np.random.seed(42)
    true_list = []
    pred_list = []
    scene_names = []
    # 打开并读取文件
    with open('../score.txt', 'r') as file:
        for line in file:
            # 分割每行内容
            parts = line.strip().split()
            scene_names.append(parts[0].split("_")[0])  # 保存场景名称
            # 提取真实值和预测值，并转换为浮点数
            true_value = float(parts[1])
            pred_value = float(parts[2])
            # 添加到对应的列表中
            true_list.append(true_value)
            pred_list.append(pred_value)

        # 转换为NumPy数组
        true_array = np.array(true_list)
        pred_array = np.array(pred_list)
    x_full = true_array
    y_full = pred_array

    # 计算分块参数
    n_chunks = data_length // chunk_size
    cols = 6  # 每行显示6个子图
    rows = n_chunks // cols + (1 if n_chunks % cols != 0 else 0)

    # 创建大图框架
    fig = plt.figure(figsize=(24, 2*rows), dpi=300)
    gs = fig.add_gridspec(rows, cols, wspace=0.4, hspace=0.6,
                          left=0.05, right=0.95,
                          bottom=0.05, top=0.95)

    # 循环绘制每个分块
    for i in range(n_chunks):
        row = i // cols
        col = i % cols

        # 获取分块数据
        start = i * chunk_size
        end = start + chunk_size
        x_chunk = x_full[start:end]
        y_chunk = y_full[start:end]
        name = scene_names[start]

        # 创建子图并绘制
        ax = fig.add_subplot(gs[row, col])
        plot_chunk(ax, x_chunk, y_chunk, i,name)

    # 保存并显示结果
    plt.tight_layout()
    # 保存无白边图片
    plt.savefig('combined_plots.png', bbox_inches='tight', pad_inches=0)
    plt.show()

# 执行拼接函数
create_mosaic()