import pandas as pd
import matplotlib

import csv
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns;sns.set()

# config font type
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#! 之后改到3000、4000个iteration会好看一点，现在这个曲线不好看

# read the data
stack_data = pd.read_csv("/home/ustc/Documents/my paper/iros2024_tex/plot/data/TaskRewardAblation_steppingstone.csv")

# smoothing factor  #! smoothing factor也要改小一点，现在曲线看上去太光滑了
TSBOARD_SMOOTHING = 0.85
# TSBOARD_SMOOTHING = 0.7

# apply exponential moving average smoothingto the data
stack_data   = stack_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

# convert to numpy array. shape: (3000epoch, 10)
stack_data   = np.array(stack_data)

#  Reward Mean - Reward Std, Rewad Mean, Reward Mean + Reward Std
#! Normal
stack_data0 = np.concatenate((stack_data[:,0]-stack_data[:,1],stack_data[:,0],stack_data[:,0] + stack_data[:,1]))   # shape: (9000,)
#! NoExploration
stack_data1 = np.concatenate((stack_data[:,2]-stack_data[:,3],stack_data[:,2],stack_data[:,2] + stack_data[:,3]))
#! NoParkour
stack_data2 = np.concatenate((stack_data[:,4]-stack_data[:,5],stack_data[:,4],stack_data[:,4] + stack_data[:,5]))
#! NoNormalization
stack_data3 = np.concatenate((stack_data[:,6]-stack_data[:,7],stack_data[:,6],stack_data[:,6] + stack_data[:,7]))

# aggregate the calculated data into a list
stack_data_total = []
stack_data_total.append(stack_data0)
stack_data_total.append(stack_data1)
stack_data_total.append(stack_data2)
stack_data_total.append(stack_data3)
print("stack_data_total: ",stack_data_total)

# prepare x-axis data representing ecoch
stack_epoch1 = range(len(stack_data[:,0]))  # range(0, 5999)
stack_epoch2 = range(len(stack_data[:,0]))
stack_epoch3 = range(len(stack_data[:,0]))
stack_epoch = np.concatenate((stack_epoch1,stack_epoch2,stack_epoch3))
# nut_epoch = stack_epoch # array([   0,    1,    2, ..., 2997, 2998, 2999])


labels = [
    'LEEPS',
    'LEEPS w/o Exploration',
    'LEEPS w/o Parkour',
    'LEEPS w/o Normalization',
]

color1 = sns.color_palette('deep')  # 比较鲜亮且深沉, 用来表示自己的方法
color2 = sns.color_palette('muted') # 更柔和、更少饱和度的颜色，用来表示baseline
color3=[
        color1[2],
        color2[3],
        color2[0],
        color2[4],
       ]

# Initialize the figure
# lgd = plt.figure(figsize=(16/4*3, 9/4*3))
lgd = plt.figure(figsize=(4/4*3, 3/4*3))    #! 太空了，调整比例

# create a subplot
stack = plt.subplot(221)    # subplot的格式是（行，列，第几个图）
for i in range(len(stack_data_total)):  # 5条线
    print(i)
    stack = sns.lineplot(x=stack_epoch, y=stack_data_total[i],label=labels[i], color=color3[i]) #! plot the data

# customize the x-axis ticks
# plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000], [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# set axis limit
# stack.set_xlim(0,)
stack.set_ylim(-0.1, )  # 调整y轴的范围

# set labels and title for the subplot
plt.ylabel("Task Reward", fontsize=14)
plt.title("Discrete Rough", fontsize=16)    #! label的字需要变大
print('finsh Discrete Rough')
stack.legend_.remove()  # 把subplot里的legend(label)去掉

labels = [
    'LEEPS',
    'LEEPS w/o Exploration',
    'LEEPS w/o Parkour',
    'LEEPS w/o Normalization',
]

print('lgd.axes is :', lgd.axes)
lines_labels = [ax.get_legend_handles_labels() for ax in lgd.axes]
print('lines_labels are:', lines_labels)
lines, label = [sum(lol, []) for lol in zip(*lines_labels)]

# 把label加到low center
leg = lgd.legend([lines[0], lines[1], lines[2], lines[3], ], labels,
                 loc='lower center',
                 ncol=5,
                 borderaxespad=-0.65,
                 frameon=False,
                 fontsize=14
                 )

for line in leg.get_lines():
    line.set_linewidth(4)   # 修改label中的线宽

# Adjust the layout of the plot to fit everything neatly
plt.tight_layout()

# plt.savefig('plot_HyTL.pdf',
#             format='pdf',
#             bbox_extra_artists=(lgd,),
#             bbox_inches='tight',
#             )
# plt.savefig('plot_HyTL.png',
#             dpi=500,
#             format='png',
#             bbox_extra_artists=(lgd,),
#             bbox_inches='tight')

print('save done')

plt.show()