import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines

# Manually entered data
data = {
    "No/Yes White Noise (0, 1)": [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    "Stabilized Baseline SysBP": [62, 57, 59, 71, 61, 92, 80, 79, 78, 78, 77, 92],
    "SysBP POST": [83, 71, 67, 81, 75, 102, 98, 85, 79, 87, 84, 97]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Define the colors for the white noise condition
colors = {0: 'blue', 1: 'red'}

# Create a figure and a single subplot
fig, ax = plt.subplots(figsize=(8, 6))

# Numeric x-coordinates
x_baseline = 1
x_post = 1.01  # Adjust this value to control the gap

# Scatter plot and line plot within the loop
for i in df.index:
    participant = df.iloc[i]
    ax.scatter([x_baseline, x_post], 
               [participant['Stabilized Baseline SysBP'], participant['SysBP POST']], 
               color=colors[participant['No/Yes White Noise (0, 1)']])
    ax.plot([x_baseline, x_post], 
            [participant['Stabilized Baseline SysBP'], participant['SysBP POST']], 
            color=colors[participant['No/Yes White Noise (0, 1)']])

# Set plot title and labels
# Set custom x-ticks and labels
    
ax.set_xticks([x_baseline, x_post])
ax.set_xticklabels(['Baseline', 'Post'])
ax.set_xlabel('Experiment Time Point', fontsize=20)
ax.set_ylabel('Heart Rate (bpm)', fontsize=25)

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=14)


# Create legend handles
blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=10, label='No White Noise')
red_line = mlines.Line2D([], [], color='red', marker='o',
                         markersize=10, label='White Noise')

# Add the legend to the plot
ax.legend(handles=[blue_line, red_line], loc='upper center', fontsize=10)

# Show the plot
plt.show()