import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ["0.0557", "0.0267", "0.3423", "0.4425", "0.0982", "0.0131", "0.0214"]
values = [0.0557, 0.0267, 0.3423, 0.4425, 0.0982, 0.0131, 0.0214]

# 创建图形和轴
fig, ax = plt.subplots()

# 找到最高值的索引
max_value_index = np.argmax(values)

# 设置颜色
colors = ['lightgreen' if i != max_value_index else 'green' for i in range(len(values))]

# 绘制条形图
bars = ax.bar(labels, values, color=colors)

# 设置透明度
for i, bar in enumerate(bars):
    bar.set_alpha(0.5if i == max_value_index else 0.3)  # 最高值不透明，其他透明

# 添加数据标签
for bar, value in zip(bars, values):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{value:.4f}', ha='center', va='bottom', color='green')

# 添加水平线
ax.hlines(0, -0.5, len(labels)-0.5, color='green', linewidth=3)

# 隐藏横纵标
ax.axis('off')

# 显示图形
plt.show()