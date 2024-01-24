import matplotlib.pyplot as plt
import numpy as np

# 假设数据 - 替换为你的实际数据
steps = np.linspace(0, 3, 300)  # 0到3百万步，300个数据点
reward_stack = np.random.normal(loc=0, scale=1, size=steps.shape).cumsum() + steps**2
reward_cleanup = np.random.normal(loc=0, scale=1, size=steps.shape).cumsum() + steps**2

# 创建图表
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2行1列的子图

# 绘制 Stack 图
axs[0].plot(steps, reward_stack, label='Stack')
axs[0].fill_between(steps, reward_stack - reward_stack.std(), reward_stack + reward_stack.std(), alpha=0.2)
axs[0].set_title('Stack')

# 绘制 Cleanup 图
axs[1].plot(steps, reward_cleanup, label='Cleanup')
axs[1].fill_between(steps, reward_cleanup - reward_cleanup.std(), reward_cleanup + reward_cleanup.std(), alpha=0.2)
axs[1].set_title('Cleanup')

# 设置标签和图例
for ax in axs:
    ax.set_xlabel('Million Steps')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
