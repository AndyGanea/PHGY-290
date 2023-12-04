import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import pymc3 as pm
from theano import shared

# Manually input the data into the DataFrame
# Replace the ellipses with the actual data
data = {
    'SUDS Score Pre': [30, 60, 40, 10, 10, 20, 20, 30, 50, 30, 10, 30],  # Replace with actual SUDS Score Pre data
    'Sleep Levels': [4, 2, 4, 3, 2, 3, 2, 4, 5, 5, 4, 2],  # Replace with actual Sleep Levels data
    'Age': [19, 19, 19, 19, 18, 20, 20, 19, 20, 20, 19, 20],  # Replace with actual Age data
    'Sex': [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # Replace with actual Sex data (0 for male, 1 for female)
    'Change in SysBP': [7, 4, 4, 5, 6, 13, 16, 8, 4, 5, 2, 5],  # Replace with actual Change in SysBP data
    'Change in DiaBP': [-5, 8, -7, 5, 4, -6, -3, 5, -7, 2, -5, -3],  # Replace with actual Change in DiaBP data
    'Change in HR': [15, 9, 3, 5, 8, 4, 10, 9, -6, -2, -3, -5],  # Replace with actual Change in HR data
    'Treatment': [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0]  # Replace with actual binary treatment data (0 for control, 1 for treatment)
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Assume these are the mean values from your Bayesian logistic regression model
coef_means = np.array([1.31, 2.90, 6.13, 2.12, -19.36, -4.87, -1.25])
intercept_mean = np.array([0.37])

# Separate the features from the treatment variable
X = df.drop('Treatment', axis=1)
y = df['Treatment'].astype(int)

## Fit the logistic regression model
logit_model = LogisticRegression()
logit_model.fit(X, y)

# Calculate the propensity scores
propensity_scores = logit_model.predict_proba(X)[:, 1]

# Create two sets of indices for treatment and control groups
treatment_indices = y[y == 1].index
control_indices = y[y == 0].index

# Fit nearest neighbors model on control group propensity scores
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(propensity_scores[control_indices].reshape(-1, 1))

print("Propensity Scores for Treatment Group:", propensity_scores[treatment_indices])
print("Propensity Scores for Control Group:", propensity_scores[control_indices])

# Calculate the pairwise distances between all treatment and control propensity scores
dist_matrix = pairwise_distances(propensity_scores[treatment_indices].reshape(-1, 1),
                                 propensity_scores[control_indices].reshape(-1, 1))

matched_treatment = []
matched_control = []

# Iterate through each treatment participant
for t_index in treatment_indices:
    # Adjust t_index to access the correct row in dist_matrix
    t_index_adjusted = np.where(treatment_indices == t_index)[0][0]

    # Find the control participant with the closest propensity score that hasn't been matched yet
    closest_controls = np.argsort(dist_matrix[t_index_adjusted])

    for c_index in closest_controls:
        control_index = control_indices[c_index]
        if control_index not in matched_control:
            matched_treatment.append(t_index)
            matched_control.append(control_index)
            break

# Form the matched pairs
matched_pairs = list(zip(matched_treatment, matched_control))

# Output the matched pairs
for pair in matched_pairs:
    print(f"Treatment participant {pair[0] + 1} is matched with Control participant {pair[1] + 1}")





# Separate the features from the treatment variable
X_shared = shared(df.drop('Treatment', axis=1).values)
y_shared = shared(df['Treatment'].values)

with pm.Model() as logistic_model:
    # Priors for the model parameters
    intercept = pm.Normal('Intercept', mu=0, sd=10)
    slopes = pm.Normal('Slopes', mu=0, sd=5, shape=X_shared.get_value().shape[1])

    # Expected value of outcome (Bernoulli likelihood)
    logits = intercept + pm.math.dot(X_shared, slopes)
    likelihood = pm.Bernoulli('y', logit_p=logits, observed=y_shared)
    
    # Sample from the posterior
    trace = pm.sample(1000, tune=2000, cores=1)  # cores=1 to avoid multiprocessing issues

# Extract the mean coefficient values from the trace
coef_means = np.mean(trace['Slopes'], axis=0)
intercept_mean = np.mean(trace['Intercept'])

# Calculate propensity scores using the mean coefficients
propensity_scores_bayesian = 1 / (1 + np.exp(-(np.dot(X_shared.get_value(), coef_means) + intercept_mean)))

# Create two sets of indices for treatment and control groups
treatment_indices = df[df['Treatment'] == 1].index
control_indices = df[df['Treatment'] == 0].index

# Fit nearest neighbors model on control group propensity scores
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(propensity_scores_bayesian[control_indices].reshape(-1, 1))

print("Propensity Scores for Treatment Group:", propensity_scores_bayesian[treatment_indices])
print("Propensity Scores for Control Group:", propensity_scores_bayesian[control_indices])

dist_matrix = pairwise_distances(propensity_scores_bayesian[treatment_indices].reshape(-1, 1),
                                 propensity_scores_bayesian[control_indices].reshape(-1, 1))

# Initialize lists to hold the matches
matched_treatment = []
matched_control = []

# Iterate through each treatment participant
for t_index in treatment_indices:
    # Adjust t_index to access the correct row in dist_matrix
    t_index_adjusted = np.where(treatment_indices == t_index)[0][0]

    # Find the control participant with the closest propensity score that hasn't been matched yet
    closest_controls = np.argsort(dist_matrix[t_index_adjusted])

    for c_index in closest_controls:
        control_index = control_indices[c_index]
        if control_index not in matched_control:
            matched_treatment.append(t_index)
            matched_control.append(control_index)
            break

# Form the matched pairs
matched_pairs = list(zip(matched_treatment, matched_control))

# Output the matched pairs
for pair in matched_pairs:
    print(f"Treatment participant {pair[0] + 1} is matched with Control participant {pair[1] + 1}")






# Extract the mean coefficient values from the Bayesian trace
coef_means = np.mean(trace['Slopes'], axis=0)
intercept_mean = np.mean(trace['Intercept'])

# Define the extent of Bayesian influence (between 0 and 1)
bayesian_influence = 0.15  # Adjust this to control the influence

coef_variances = np.var(trace['Slopes'], axis=0)

# Calculate the weights as the inverse of the variance
weights = 1 / coef_variances

# Normalize the weights so they sum to 1
weights_normalized = (weights / np.sum(weights)) * bayesian_influence

# Fit the logistic regression model
logistic_regression_weight = 1 - np.sum(weights_normalized)
logit_model = LogisticRegression()
logit_model.fit(X, y)

# Compute the weighted average of the coefficients
bayesian_coefs_weighted = coef_means * weights_normalized
logistic_coefs_weighted = logit_model.coef_[0] * logistic_regression_weight

# Use the mean of the Bayesian slopes for the adjustment
adjusted_coefs = bayesian_coefs_weighted + logistic_coefs_weighted

# Calculate adjusted logits
adjusted_logits = np.dot(X, adjusted_coefs.T) + logit_model.intercept_

# Flatten adjusted_logits if it's two-dimensional
if adjusted_logits.ndim > 1:
    adjusted_logits = adjusted_logits.flatten()


# Convert adjusted logits to probabilities using the logistic function
propensity_scores_adjusted = 1 / (1 + np.exp(-adjusted_logits))

# Create two sets of indices for treatment and control groups
treatment_indices = df[df['Treatment'] == 1].index
control_indices = df[df['Treatment'] == 0].index

# Fit nearest neighbors model on control group propensity scores
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(propensity_scores_adjusted[control_indices].reshape(-1, 1))

print("Propensity Scores for Treatment Group:", propensity_scores_adjusted[treatment_indices])
print("Propensity Scores for Control Group:", propensity_scores_adjusted[control_indices])

dist_matrix = pairwise_distances(propensity_scores_adjusted[treatment_indices].reshape(-1, 1),
                                 propensity_scores_adjusted[control_indices].reshape(-1, 1))

# Initialize lists to hold the matches
matched_treatment = []
matched_control = []

# Iterate through each treatment participant
for t_index in treatment_indices:
    # Adjust t_index to access the correct row in dist_matrix
    t_index_adjusted = np.where(treatment_indices == t_index)[0][0]

    # Find the control participant with the closest propensity score that hasn't been matched yet
    closest_controls = np.argsort(dist_matrix[t_index_adjusted])

    for c_index in closest_controls:
        control_index = control_indices[c_index]
        if control_index not in matched_control:
            matched_treatment.append(t_index)
            matched_control.append(control_index)
            break

# Form the matched pairs
matched_pairs = list(zip(matched_treatment, matched_control))

# Output the matched pairs
for pair in matched_pairs:
    print(f"Treatment participant {pair[0] + 1} is matched with Control participant {pair[1] + 1}")