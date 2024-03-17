import matplotlib.pyplot as plt
import numpy as np

# Sample data: Replace these with your actual data points
environments = ['2k steps in Log Bridge', '6k steps in Log Bridge', '2k steps in Tunnel', '6k steps in Tunnel']
VelocityTracking_scores = [0.8, 0.9, 0.7, 0.6]
ExtremeParkour_scores = [0.9, 0.8, 0.8, 0.7]
Oracle_scores = [0.7, 0.8, 0.6, 0.5]
LEEPS_scores = [0.6, 0.7, 0.5, 0.4]

# Set the positions and width for the bars
positions = np.arange(len(environments))
width = 0.2  # the width of the bars

# Plot the bars
fig, ax = plt.subplots()
rects1 = ax.bar(positions - width*1.5, VelocityTracking_scores, width, label='Velocity Tracking')
rects2 = ax.bar(positions - width/2, ExtremeParkour_scores, width, label='Extreme Parkour')
rects3 = ax.bar(positions + width/2, Oracle_scores, width, label='LEEPS Oracle')
rects4 = ax.bar(positions + width*1.5, LEEPS_scores, width, label='LEEPS')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Terrain Level')
ax.set_title('Scores by environment')
ax.set_xticks(positions)
ax.set_xticklabels(environments)
ax.legend()

# Function to add labels on top of the bars
def add_labels(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call the function to add labels
add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)

# Display the plot
plt.tight_layout()
plt.show()
