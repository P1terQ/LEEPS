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
discrete_data = pd.read_csv("/home/ustc/Documents/my paper/iros2024_tex/plot/data/TaskRewardAblation_discrete.csv")
step_data = pd.read_csv("/home/ustc/Documents/my paper/iros2024_tex/plot/data/TaskRewardAblation_step.csv")
steppingstone_data = pd.read_csv("/home/ustc/Documents/my paper/iros2024_tex/plot/data/TaskRewardAblation_steppingstone1.csv")
gap_data = pd.read_csv("/home/ustc/Documents/my paper/iros2024_tex/plot/data/TaskRewardAblation_gap.csv")

# smoothing factor  #! smoothing factor也要改小一点，现在曲线看上去太光滑了
TSBOARD_SMOOTHING = 0.8
# TSBOARD_SMOOTHING = 0.7

# apply exponential moving average smoothingto the data
discrete_data = discrete_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
step_data = step_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
steppingstone_data = steppingstone_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
gap_data = gap_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

# convert to numpy array. shape: (3000epoch, 10)
discrete_data = np.array(discrete_data)
step_data = np.array(step_data)
steppingstone_data = np.array(steppingstone_data)
gap_data = np.array(gap_data)

NUM_EPOCHS = 4000

#  Reward Mean - Reward Std, Rewad Mean, Reward Mean + Reward Std
#! Normal
discrete_data0 = np.concatenate((discrete_data[:NUM_EPOCHS,0]-discrete_data[:NUM_EPOCHS,1],discrete_data[:NUM_EPOCHS,0],discrete_data[:NUM_EPOCHS,0] + discrete_data[:NUM_EPOCHS,1]))   # shape: (9000,)
#! NoExploration
discrete_data1 = np.concatenate((discrete_data[:NUM_EPOCHS,2]-discrete_data[:NUM_EPOCHS,3],discrete_data[:NUM_EPOCHS,2],discrete_data[:NUM_EPOCHS,2] + discrete_data[:NUM_EPOCHS,3]))
#! NoParkour
discrete_data2 = np.concatenate((discrete_data[:NUM_EPOCHS,4]-discrete_data[:NUM_EPOCHS,5],discrete_data[:NUM_EPOCHS,4],discrete_data[:NUM_EPOCHS,4] + discrete_data[:NUM_EPOCHS,5]))
#! NoNormalization
discrete_data3 = np.concatenate((discrete_data[:NUM_EPOCHS,6]-discrete_data[:NUM_EPOCHS,7],discrete_data[:NUM_EPOCHS,6],discrete_data[:NUM_EPOCHS,6] + discrete_data[:NUM_EPOCHS,7]))

NUM_EPOCHS_steps = 3000
#! Normal
step_data0 = np.concatenate((step_data[:NUM_EPOCHS_steps,0]-step_data[:NUM_EPOCHS_steps,1],step_data[:NUM_EPOCHS_steps,0],step_data[:NUM_EPOCHS_steps,0] + step_data[:NUM_EPOCHS_steps,1]))   # shape: (9000,)
#! NoExploration
step_data1 = np.concatenate((step_data[:NUM_EPOCHS_steps,2]-step_data[:NUM_EPOCHS_steps,3],step_data[:NUM_EPOCHS_steps,2],step_data[:NUM_EPOCHS_steps,2] + step_data[:NUM_EPOCHS_steps,3]))
#! NoParkour
step_data2 = np.concatenate((step_data[:NUM_EPOCHS_steps,4]-step_data[:NUM_EPOCHS_steps,5],step_data[:NUM_EPOCHS_steps,4],step_data[:NUM_EPOCHS_steps,4] + step_data[:NUM_EPOCHS_steps,5]))
#! NoNormalization
step_data3 = np.concatenate((step_data[:NUM_EPOCHS_steps,6]-step_data[:NUM_EPOCHS_steps,7],step_data[:NUM_EPOCHS_steps,6],step_data[:NUM_EPOCHS_steps,6] + step_data[:NUM_EPOCHS_steps,7]))

NUM_EPOCHS_steppingstone = 5000
#! Normal
steppingstone_data0 = np.concatenate((steppingstone_data[:NUM_EPOCHS_steppingstone,0]-steppingstone_data[:NUM_EPOCHS_steppingstone,1],steppingstone_data[:NUM_EPOCHS_steppingstone,0],steppingstone_data[:NUM_EPOCHS_steppingstone,0] + steppingstone_data[:NUM_EPOCHS_steppingstone,1]))   # shape: (9000,)
#! NoExploration
steppingstone_data1 = np.concatenate((steppingstone_data[:NUM_EPOCHS_steppingstone,2]-steppingstone_data[:NUM_EPOCHS_steppingstone,3],steppingstone_data[:NUM_EPOCHS_steppingstone,2],steppingstone_data[:NUM_EPOCHS_steppingstone,2] + steppingstone_data[:NUM_EPOCHS_steppingstone,3]))
#! NoParkour
steppingstone_data2 = np.concatenate((steppingstone_data[:NUM_EPOCHS_steppingstone,4]-steppingstone_data[:NUM_EPOCHS_steppingstone,5],steppingstone_data[:NUM_EPOCHS_steppingstone,4],steppingstone_data[:NUM_EPOCHS_steppingstone,4] + steppingstone_data[:NUM_EPOCHS_steppingstone,5]))
#! NoNormalization
steppingstone_data3 = np.concatenate((steppingstone_data[:NUM_EPOCHS_steppingstone,6]-steppingstone_data[:NUM_EPOCHS_steppingstone,7],steppingstone_data[:NUM_EPOCHS_steppingstone,6],steppingstone_data[:NUM_EPOCHS_steppingstone,6] + steppingstone_data[:NUM_EPOCHS_steppingstone,7]))

#! Normal
gap_data0 = np.concatenate((gap_data[:NUM_EPOCHS,0]-gap_data[:NUM_EPOCHS,1],gap_data[:NUM_EPOCHS,0],gap_data[:NUM_EPOCHS,0] + gap_data[:NUM_EPOCHS,1]))   # shape: (9000,)
#! NoExploration
gap_data1 = np.concatenate((gap_data[:NUM_EPOCHS,2]-gap_data[:NUM_EPOCHS,3],gap_data[:NUM_EPOCHS,2],gap_data[:NUM_EPOCHS,2] + gap_data[:NUM_EPOCHS,3]))
#! NoParkour
gap_data2 = np.concatenate((gap_data[:NUM_EPOCHS,4]-gap_data[:NUM_EPOCHS,5],gap_data[:NUM_EPOCHS,4],gap_data[:NUM_EPOCHS,4] + gap_data[:NUM_EPOCHS,5]))
#! NoNormalization
gap_data3 = np.concatenate((gap_data[:NUM_EPOCHS,6]-gap_data[:NUM_EPOCHS,7],gap_data[:NUM_EPOCHS,6],gap_data[:NUM_EPOCHS,6] + gap_data[:NUM_EPOCHS,7]))

# aggregate the calculated data into a list
discrete_data_total = []
discrete_data_total.append(discrete_data0)
discrete_data_total.append(discrete_data1)
discrete_data_total.append(discrete_data2)
discrete_data_total.append(discrete_data3)


step_data_total = []
step_data_total.append(step_data0)
step_data_total.append(step_data1)
step_data_total.append(step_data2)
step_data_total.append(step_data3)
print("step_data_total: ",step_data_total)

steppingstone_data_total = []
steppingstone_data_total.append(steppingstone_data0)
steppingstone_data_total.append(steppingstone_data1)
steppingstone_data_total.append(steppingstone_data2)
steppingstone_data_total.append(steppingstone_data3)
print("steppingstone_data_total: ",steppingstone_data_total)

gap_data_total = []
gap_data_total.append(gap_data0)
gap_data_total.append(gap_data1)
gap_data_total.append(gap_data2)
gap_data_total.append(gap_data3)
print("gap_data_total: ",gap_data_total)

# prepare x-axis data representing ecoch
epoch1 = range(NUM_EPOCHS)  # range(0, 5999)
epoch2 = range(NUM_EPOCHS)
epoch3 = range(NUM_EPOCHS)
epoch = np.concatenate((epoch1,epoch2,epoch3))
print("epoch shape: ",epoch.shape)

epoch1_steps = range(NUM_EPOCHS_steps)  # range(0, 5999)
epoch2_steps = range(NUM_EPOCHS_steps)
epoch3_steps = range(NUM_EPOCHS_steps)
epoch_steps = np.concatenate((epoch1_steps,epoch2_steps,epoch3_steps))

epoch1_steppingstone = range(NUM_EPOCHS_steppingstone)  # range(0, 5999)
epoch2_steppingstone = range(NUM_EPOCHS_steppingstone)
epoch3_steppingstone = range(NUM_EPOCHS_steppingstone)
epoch_steppingstone = np.concatenate((epoch1_steppingstone,epoch2_steppingstone,epoch3_steppingstone))

labels = [
    'LEEPS',
    'LEEPS w/o Exploration',
    'LEEPS w/o Parkour',
    'LEEPS w/o Normalization',
]

color1 = sns.color_palette('deep')  # 比较鲜亮且深沉, 用来表示自己的方法
color2 = sns.color_palette('muted') # 更柔和、更少饱和度的颜色，用来表示baseline
color3=[
        color2[3],
        color1[2],
        color2[0],
        color2[4],
       ]

# Initialize the figure
# lgd = plt.figure(figsize=(16/4*3, 9/4*3))
lgd = plt.figure(figsize=(20, 5.5))    #! 太空了，调整比例

# create a subplot
stack = plt.subplot(141)    # subplot的格式是（行，列，第几个图）
for i in range(len(discrete_data_total)):  # 5条线
    print(i)
    stack = sns.lineplot(x=epoch, y=discrete_data_total[i],label=labels[i], color=color3[i]) #! plot the data

# customize the x-axis ticks
plt.xticks([0, 1000, 2000, 3000, 4000], [0, 1.0, 2.0, 3.0, 4.0])
# set axis limit
stack.set_ylim(-0.1, )  # 调整y轴的范围

# set labels and title for the subplot
plt.ylabel("Task Reward", fontsize=16)
plt.xlabel("Thousand Steps", fontsize=14)
plt.title("Discrete Rough", fontsize=18)    #! label的字需要变大
print('finsh Discrete Rough')
stack.legend_.remove()  # 把subplot里的legend(label)去掉


# create a subplot
stack = plt.subplot(142)    # subplot的格式是（行，列，第几个图）
for i in range(len(step_data_total)):  # 5条线
    print(i)
    stack = sns.lineplot(x=epoch_steps, y=step_data_total[i],label=labels[i], color=color3[i]) #! plot the data

# customize the x-axis ticks
plt.xticks([0, 1000, 2000, 3000], [0, 1.0, 2.0, 3.0])
# set axis limit
stack.set_ylim(-0.1, )  # 调整y轴的范围

# set labels and title for the subplot
# plt.ylabel("Task Reward", fontsize=16)
plt.xlabel("Thousand Steps", fontsize=14)
plt.title("Step", fontsize=18)    #! label的字需要变大
print('finsh Step')
stack.legend_.remove()

# create a subplot
stack = plt.subplot(143)    # subplot的格式是（行，列，第几个图）
for i in range(len(steppingstone_data_total)):  # 5条线
    print(i)
    stack = sns.lineplot(x=epoch_steppingstone, y=steppingstone_data_total[i],label=labels[i], color=color3[i]) #! plot the data

# customize the x-axis ticks
plt.xticks([0, 1000, 2000, 3000, 4000, 5000], [0, 1.0, 2.0, 3.0, 4.0, 5.0])
# set axis limit
stack.set_ylim(-0.1, )  # 调整y轴的范围

# set labels and title for the subplot
plt.xlabel("Thousand Steps", fontsize=14)
# plt.ylabel("Task Reward", fontsize=16)
plt.title("Stepping stones", fontsize=18)    #! label的字需要变大
print('finsh Stepping stones')
stack.legend_.remove()

# create a subplot
stack = plt.subplot(144)    # subplot的格式是（行，列，第几个图）
for i in range(len(gap_data_total)):  # 5条线
    print(i)
    stack = sns.lineplot(x=epoch, y=gap_data_total[i],label=labels[i], color=color3[i]) #! plot the data

# customize the x-axis ticks
plt.xticks([0, 1000, 2000, 3000, 4000], [0, 1.0, 2.0, 3.0, 4.0])
# set axis limit
stack.set_ylim(-0.1, )  # 调整y轴的范围

# set labels and title for the subplot
plt.xlabel("Thousand Steps", fontsize=14)
# plt.ylabel("Task Reward", fontsize=16)
plt.title("Gap", fontsize=18)    #! label的字需要变大
print('finsh Gap')
stack.legend_.remove()


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
                 borderaxespad=-0.45, #-0.65,
                 frameon=False,
                 fontsize=15
                 )

for line in leg.get_lines():
    line.set_linewidth(4)   # 修改label中的线宽

# Adjust the layout of the plot to fit everything neatly
plt.tight_layout()

plt.savefig('plot_RewardAblation.pdf',
            format='pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
            )
plt.savefig('plot_RewardAblation.png',
            dpi=500,
            format='png',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')

print('save done')

plt.show()