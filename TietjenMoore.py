import numpy as np
from scipy.optimize import minimize
from scipy.stats import t

# Prepare the data
data = np.array([74, 77, 101, 87, 78, 61, 59, 102, 79, 79, 76, 65, 71, 75])

# Specify the suspected outliers
specified_outliers = np.array([101, 102, 59])  # Replace these with the values you suspect are outliers

# Calculate the test statistic
remaining_data = [x for x in data if x not in specified_outliers]
SSR = np.sum((data - np.mean(data)) ** 2)
SSR_star = np.sum((remaining_data - np.mean(remaining_data)) ** 2)
N = len(data)
k = 2
T = (SSR - SSR_star) / (SSR_star / (N - k))

# Calculate the critical value
alpha = 0.05
F_critical = t.ppf(1 - alpha, k, N - k)

# Compare T and F_critical
if T > F_critical:
    print(f"The specified outliers {specified_outliers} are indeed outliers.")
else:
    print(f"The specified outliers {specified_outliers} are not outliers.")