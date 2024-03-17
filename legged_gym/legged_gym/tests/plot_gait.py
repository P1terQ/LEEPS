import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define file paths
csv_file_paths = [
    '/home/ustc/robot/projects/legged_locomotion/iros2024/legged_gym/legged_gym/tests/data/gait_vel.csv',
    '/home/ustc/robot/projects/legged_locomotion/iros2024/legged_gym/legged_gym/tests/data/gait_bad.csv',
    '/home/ustc/robot/projects/legged_locomotion/iros2024/legged_gym/legged_gym/tests/data/gait_good.csv'
]

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.8))  # Adjust the size as needed
colors = ['C0', 'C2', 'C1']
titles = ['Velocity tracking', 'LEEPS w/o gait-shaping', 'LEEPS']

title_fontsize = 20
label_fontsize = 18
padding = 0.6

pic_num = 0

#! 1st subplot
ax = axs[pic_num]
df = pd.read_csv(csv_file_paths[pic_num])

# Plot each contact state
for i, limb in enumerate(['RH', 'RF', 'LH', 'LF']):
    # Extract the times where contact is made
    contact_times = df['Time'][df[limb] == 1]
    for time in contact_times:
        # ax.plot([time, time + 0.02], [i, i], color=colors[pic_num], lw=20)  # Adjust line width as needed
        ax.plot([time, time + 0.002], [i, i], color=colors[pic_num], lw=20)

# Set the y-axis labels to the limbs names
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['LF', 'LH', 'RF', 'RH'], fontsize=label_fontsize)
ax.set_ylim([-padding, 4 - 1 + padding])
# ax.invert_yaxis()  # Invert y-axis so that LF is at the bottom as in the example image

# Set the x-axis label and limits
ax.set_xlabel('Time [s]', fontsize=label_fontsize)
ax.set_xlim([0, max(df['Time'])])
# ax.set_xlim([df['Time'].min() - padding, df['Time'].max() + padding])
ax.set_title(titles[pic_num], fontsize=title_fontsize)
pic_num += 1

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=label_fontsize, width=1.5)

#! 2ed subplot
ax = axs[pic_num]
df = pd.read_csv(csv_file_paths[pic_num])

# Plot each contact state
for i, limb in enumerate(['RH', 'RF', 'LH', 'LF']):
    # Extract the times where contact is made
    contact_times = df['Time'][df[limb] == 1]
    for time in contact_times:
        # ax.plot([time, time + 0.02], [i, i], color=colors[pic_num], lw=20)  # Adjust line width as needed
        ax.plot([time, time + 0.003], [i, i], color=colors[pic_num], lw=20)

# Set the y-axis labels to the limbs names
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['LF', 'LH', 'RF', 'RH'], fontsize=label_fontsize)
ax.set_ylim([-padding, 4 - 1 + padding])
# ax.invert_yaxis()  # Invert y-axis so that LF is at the bottom as in the example image

# Set the x-axis label and limits
ax.set_xlabel('Time [s]', fontsize=label_fontsize)
ax.set_xlim([0, max(df['Time'])])
ax.set_title(titles[pic_num], fontsize=title_fontsize)
pic_num += 1

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=label_fontsize, width=1.5)

#! 3rd subplot
ax = axs[pic_num]
df = pd.read_csv(csv_file_paths[pic_num])

# Plot each contact state
for i, limb in enumerate(['RH', 'RF', 'LH', 'LF']):
    # Extract the times where contact is made
    contact_times = df['Time'][df[limb] == 1]
    for time in contact_times:
        # ax.plot([time, time + 0.02], [i, i], color=colors[pic_num], lw=20)  # Adjust line width as needed
        ax.plot([time, time + 0.0015], [i, i], color=colors[pic_num], lw=20)

# Set the y-axis labels to the limbs names
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['LF', 'LH', 'RF', 'RH'], fontsize=label_fontsize)
ax.set_ylim([-padding, 4 - 1 + padding])

# ax.set_xticks([0, 1, 2], [0, 1.0, 2.0])
# ax.invert_yaxis()  # Invert y-axis so that LF is at the bottom as in the example image

# Set the x-axis label and limits
ax.set_xlabel('Time [s]', fontsize=label_fontsize)
ax.set_xlim([0, max(df['Time'])])
ax.set_title(titles[pic_num], fontsize=title_fontsize)
pic_num += 1

# Remove all spines
# for spine in ax.spines.values():
#     spine.set_visible(False)

# Hide the top and right ticks and tick labels
# ax.xaxis.tick_bottom()
# ax.yaxis.tick_left()

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=label_fontsize, width=1.5)

    
# Adjust layout for better spacing
plt.tight_layout()

plt.savefig('gait_phase.png', format='png', dpi=500)  # Save as PNG with specified DPI
plt.savefig('gait_phase.pdf', format='pdf')  # Save as PDF

# Show the plot
plt.show()
