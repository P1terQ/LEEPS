import matplotlib.pyplot as plt
import numpy as np

# Define the data - using binary values as an example
# You can replace these with your actual data
data1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]])
data2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]])
data3 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Define the labels for y-axis
labels = ['RH', 'RF', 'LH', 'LF']

# Create a figure with 3 subplots in a row
fig, axs = plt.subplots(1, 3, figsize=(9, 3))

# A function to create a single subplot
def create_subplot(ax, data, title, c_map):
    ax.imshow(data, cmap=c_map, aspect='auto')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time [s]')
    ax.set_title(title)

# Create each subplot with the corresponding data and color map
create_subplot(axs[0], data1, 'Data 1', 'Blues')
create_subplot(axs[1], data2, 'Data 2', 'Oranges')
create_subplot(axs[2], data3, 'Data 3', 'Greens')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
