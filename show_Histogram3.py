import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ["0.0155", "0.0412", "0.6153", "0.1785", "0.0348", "0.0055", "0.1092"]
values = [0.0155, 0.0412, 0.6153, 0.1785, 0.0348, 0.0055, 0.1092]

# 创建图形和轴
fig, ax = plt.subplots()

# 找到最高值的索引
max_value_index = np.argmax(values)

# 设置颜色
colors = ['lightpink' if i != max_value_index else 'lightpink' for i in range(len(values))]

# 绘制条形图
bars = ax.bar(labels, values, color=colors)

# 设置透明度
for i, bar in enumerate(bars):
    bar.set_alpha(1 if i == max_value_index else 0.3)  # 最高值不透明，其他透明

# 添加数据标签
for i, (bar, value) in enumerate(zip(bars, values)):
    yval = bar.get_height()
    # 检查当前值是否为最高值
    fontweight = 'bold' if i == max_value_index else 'normal'
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{value:.4f}',
            ha='center', va='bottom', color='pink', fontweight=fontweight)

# 添加水平线
ax.hlines(0, -0.5, len(labels)-0.5, color='pink', linewidth=3)

# 隐藏横纵标
ax.axis('off')
# 保存图像
plt.savefig('/data/zjl/192-torch2/VideoX-master/X-CLIP/Histogram/bar_chart.png', bbox_inches='tight', dpi=300)  # 保存为 PNG 格式，300 DPI
# 显示图形
plt.show()