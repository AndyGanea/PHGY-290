import numpy as np
from scipy import stats

# Prepare the data
data = np.array([18, 9, 8, 5, 3, 7])

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Identify the suspected outlier
suspected_outlier = 18

# Calculate Grubb's statistic
G = np.abs(suspected_outlier - mean) / std_dev

# Calculate the critical value
N = len(data)
alpha = 0.05
t_value = stats.t.ppf(1 - alpha / (2 * N), N - 2)
G_critical = ((N - 1) / np.sqrt(N)) * np.sqrt(t_value ** 2 / (N - 2 + t_value ** 2))

# Compare Grubb's statistic and critical value
if G > G_critical:
    print("The suspected data point is an outlier.")
else:
    print("The suspected data point is not an outlier.")