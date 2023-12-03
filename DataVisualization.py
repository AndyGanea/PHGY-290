# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Prepare the Data
data1 = [7979, 8687, 9984, 10030, 7878, 7416, 7644, 10212, 8464, 8580, 7931, 7344, 8051, 6675]
data2 = [9240, 9472, 10412, 16234, 10464, 7519, 9790, 11739, 8272, 10021, 8856, 8295, 7110, 10620]
# Add a third dataset
data3 = [8888, 9394, 11970, 11374, 11110, 8988, 10010, 11745, 9792, 11020, 10032, 7917, 8188, 10830]

# Create the Boxplot
box = plt.boxplot([data1, data2, data3], vert=True, patch_artist=True, showfliers=False, labels=['Baseline', 'After Test with Invisible Timer', 'After Test with Visible Timer'])  # Add a label for the third boxplot

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

# Scatter for the third dataset
y3 = data3
x3 = np.random.normal(3, 0, len(y3))  # Notice the '3' for the third dataset
plt.scatter(x3, y3, s=50, color='red', alpha=1, zorder=3)

plt.subplots_adjust(top=0.75)

# Labels and Title
plt.xlabel('Time Point in Experiment', fontsize=30)
plt.ylabel('Rate Pressure Product (mmHg * bpm)', fontsize=20)
plt.title('Rate Pressure Product (mmHg * bpm) of 14 Participants at \nBaseline, After a Visible Timer Mental Agility Test,\nand After an Invisible Timer Mental Agility Test', fontsize=40, pad=20)  # Title adjusted for length

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Set Y-Axis Range
plt.ylim(6000, 13000)

# Get Y-data of Upper Whiskers for all datasets
upper_whisker1 = box['whiskers'][1].get_ydata()[1]
upper_whisker2 = box['whiskers'][3].get_ydata()[1]
upper_whisker3 = box['whiskers'][5].get_ydata()[1]  # The upper whisker for the third dataset

# Calculate Line Heights (above the highest upper whisker)
line_height1 = max(upper_whisker1, upper_whisker3) + 1  # Adjusted line heights for each comparison
line_height2 = max(upper_whisker1, upper_whisker2) + 2
line_height3 = max(upper_whisker2, upper_whisker3) + 3

# Adjust the y-coordinates for significance bars to ensure they do not overlap
line_heights = [line_height1, line_height2, line_height3]
line_heights.sort(reverse=True)

# Show the Plot
plt.show()
