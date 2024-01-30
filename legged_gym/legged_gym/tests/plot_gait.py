import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load some example data, replace this with the actual CSV file path
# Here we create a DataFrame manually for illustration purposes
csv_file_path = '/home/ustc/robot/projects/legged_locomotion/iros2024/legged_gym/legged_gym/tests/data/gait.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)
# Define the plot
fig, ax = plt.subplots(figsize=(6, 3))

# Plot each contact state
for i, limb in enumerate(['RH', 'RF', 'LH', 'LF']):
    # Extract the times where contact is made
    contact_times = df['Time'][df[limb] == 1]
    for time in contact_times:
        ax.plot([time, time + 0.02], [i, i], color='C0', lw=10)  # Adjust line width (lw) as needed

# Set the y-axis labels to the limbs names
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['LF', 'LH', 'RF', 'RH'])
ax.invert_yaxis()  # Invert y-axis so that LF is at the bottom as in the example image

# Set the x-axis label and limits
ax.set_xlabel('Time [s]')
ax.set_xlim([0, max(df['Time'])])

# Remove all spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Hide the top and right ticks and tick labels
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# Show the plot
plt.tight_layout()
plt.show()
