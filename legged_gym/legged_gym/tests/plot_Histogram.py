import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from matplotlib import style
import numpy as np
import seaborn as sns
style.use('ggplot')  # 加载'ggplot'风格
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 加载中文字体
# font_path = "/System/Library/Fonts/STHeiti Light.ttc" # 本地字体链接
# prop = mfm.FontProperties(fname=font_path)

A = [[0.4, 0.6, 0.02, 0.9],
     [0.6, 0.8, 1.0, 0.02],
     [1.0, 1.0, 1.0, 1.0],
     [1.0, 1.0, 0.95, 0.9]]

x_labels = ['0.5m barrier', '0.8m gap', '0.2m log', '0.2m tunnel']

x = np.arange(4)
# 生成多柱图
# lgd = plt.figure(figsize=(10,3))

fig, ax = plt.subplots(figsize=(12.5,3.5))
color1 = sns.color_palette('deep')
color2 = sns.color_palette('muted')
color3=[color2[0],color2[1],color2[4],color2[3],color1[2]]
sns.set_style('white')
bar_width = 0.2
ax.bar(x + 0.0, A[0], color=color3[4], width=bar_width, label='Robot Parkour Learning')
ax.bar(x + 0.2, A[1], color=color3[0], width=bar_width, label='Extreme Legged Parkour')
ax.bar(x + 0.4, A[2], color=color3[2], width=bar_width, label='LEEPS$_\mathrm{Oracle}$')
ax.bar(x + 0.6, A[3], color=color3[3], width=bar_width, label='LEEPS')

plt.xticks(x + 0.30, x_labels, fontsize=13, color='black')
plt.xlabel('Environments',fontsize=16,color='black')
plt.ylim( -0.05 ,1.19)
plt.yticks(fontsize=12,color='black')
plt.ylabel('Success Rate', fontsize=14,color='black')
# lgd.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# 生成图片
plt.legend(fontsize=15,
           loc='upper center',
           borderaxespad=0.2,
           frameon=False,
           # loc="upper right",
           ncol=4)

plt.tight_layout()
plt.savefig("TerrainLevel.png", dpi=700)
plt.savefig('TerrainLevel.pdf',
            format='pdf',
            bbox_extra_artists=(fig,),
            bbox_inches='tight',
            )
plt.show()