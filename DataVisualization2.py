# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Prepare the Data
data1 = [18, 9, 8, 5, 3, 7]
data2 = [21, 14, 6, 10, 10, 18]

# Create the Boxplot
box = plt.boxplot([data1, data2], vert=True, patch_artist=True, showfliers=False, labels=['White Noise', 'No White Noise'])

# Make the boxes transparent by setting alpha value and adjust transparency
for patch in box['boxes']:
    patch.set_facecolor('blue')
    patch.set_alpha(0.3)

for median in box['medians']:
    median.set(linewidth=2, color='red')

# Add Scatter Plot Data Over the Boxplots
# Scatter for the first dataset
y1 = data1
x1 = np.random.normal(1, 0, len(y1))
plt.scatter(x1, y1, s=50, color='red', alpha=1, zorder=3)

# Scatter for the second dataset
y2 = data2
x2 = np.random.normal(2, 0, len(y2))
plt.scatter(x2, y2, s=50, color='red', alpha=1, zorder=3)

# plt.subplots_adjust(top=0.5)

# Labels and Title
plt.xlabel('Treatment Group', fontsize=30)
plt.ylabel('Change in Heart Rate (bpm)', fontsize=30)
# plt.title('Test Scores of 14 Participants at in a Visible\n and Invisible Timer Mental Agility Test', fontsize=40, pad=20)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Set Y-Axis Range
plt.ylim(0, 25)

# Get Y-data of Upper Whiskers for both datasets
upper_whisker1 = box['whiskers'][1].get_ydata()[1]
upper_whisker2 = box['whiskers'][3].get_ydata()[1]

# Calculate Line Heights (above the highest upper whisker)
line_height = max(upper_whisker1, upper_whisker2) + 2

# Add Significance Lines
# plt.plot([1, 2], [line_height, line_height], color='black')  # First vs Second
# plt.plot([1, 1], [upper_whisker1, line_height], color='black')
# plt.plot([2, 2], [upper_whisker2, line_height], color='black')

# Add '*' Text for Significance
# plt.text(1.5, line_height + 0.5, '*', ha='center', fontsize=18)

# Show the Plot
plt.show()
