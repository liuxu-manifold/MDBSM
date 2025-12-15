import matplotlib.pyplot as plt
import numpy as np

# 数据准备
T = [ 0.1, 0.3, 0.5, 0.7, 0.9, 1] # T 的值
WAR = [0.7390, 0.7415, 0.7437, 0.7458, 0.7407, 0.7415]  # WAR 数据
UAR = [0.6237, 0.6343, 0.6351, 0.6374, 0.6278, 0.6325]  # UAR 数据

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制 WAR 折线图
plt.plot(T, WAR, marker='s', color='red', label='WAR', linestyle='-')

# 绘制 UAR 折线图
plt.plot(T, UAR, marker='o', color='blue', label='UAR', linestyle=':')

# 添加标签和标题
plt.xlabel(r'$\alpha$', fontsize=28)
plt.ylabel('Accuracy', fontsize=28)
# plt.title('Accuracy vs T', fontsize=16)
plt.legend(fontsize=28)
# 设置 x 轴范围
# plt.xlim(0, 1)

# 设置 x 轴刻度及字体大小
plt.xticks(T, fontsize=16)  # 增大 x 轴刻度字体

# 设置 y 轴刻度及字体大小
plt.yticks(fontsize=16)  # 增大 y 轴刻度字体
# 显示网格
plt.grid()

# 保存图像
plt.savefig('/data/zjl/192-torch2/VideoX-master/X-CLIP/accuracy_vs_T.png')

# 显示图像
plt.show()