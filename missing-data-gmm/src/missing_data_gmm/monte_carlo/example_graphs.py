import matplotlib.pyplot as plt
import numpy as np

# Adjusted code: Use a single z variable for simpler visualization
rng = np.random.default_rng(123456)

# Parameters
n_samples = 400
shift_value = 1.0  # Shift for missing x in Example 3

# Generate z for all examples
z = rng.normal(0, 1, n_samples)
x = 1 * z + np.sqrt(10) * rng.normal(0, 1, n_samples)  # True x, gamma = 1

epsilon = np.sqrt(10) * rng.normal(0, 1, n_samples)  # Error term
y = 1 * x + 1 * z + epsilon  # True regression line. alpha = 1, beta = 1

# Example 1: MCAR (50% missing)
m1 = rng.uniform(size=n_samples) < 0.5  # Approximately 50% missing
# x1_obs = np.where(m1, np.nan, x)  # Observed x with NaN for missing

# Example 2: MAR_z (~50% missing based on z)
prob_missing = 1 / (
    1 + np.exp(-0.5 * z)
)  # Logistic function of z for missingness probability
m2 = rng.uniform(size=n_samples) < prob_missing  # Missing indicator based on z
# x2_obs = np.where(m2, np.nan, x)  # Observed x with NaN for missing

# Example 3: MAR_y (missing x based on unobservables)
prob_missing = 1 / (
    1 + np.exp(-0.5 * y)
)  # Logistic function of epsilon for missingness probability
m3 = rng.uniform(size=n_samples) < prob_missing  # Missing indicator based on epsilon


# Example 4: MNAR with missing x shifted (~50% missing)
m4 = rng.uniform(size=n_samples) < 0.5  # Approximately 50% missing
x4_obs = np.where(m4, x + shift_value, x)  # Shift up for missing


# Plotting and saving each graph separately

# Plot Example 1: MCAR
plt.figure(figsize=(6, 6))
plt.scatter(z[m1 == 0], x[m1 == 0], color="blue", alpha=0.6, label="Observed x")
plt.scatter(z[m1 == 1], x[m1 == 1], color="red", alpha=0.6, label="Missing x")
plt.plot(z, 1 * z, color="black", label="True Projection Line")
# plt.title('Example 1: MCAR')
plt.xlabel("z_2")
plt.ylabel("x")
plt.legend()
# example_1_path = "../../blt/graphs/example_1_mcar.png"
# plt.savefig(example_1_path)
# plt.close()

# Plot Example 2: MAR z
plt.figure(figsize=(6, 6))
plt.scatter(z[m2 == 0], x[m2 == 0], color="blue", alpha=0.6, label="Observed x")
plt.scatter(z[m2 == 1], x[m2 == 1], color="red", alpha=0.6, label="Missing x")
plt.plot(z, 1 * z, color="black", label="True Projection Line")
# plt.title('Example 2: MAR')
plt.xlabel("z_2")
plt.ylabel("x")
plt.legend()
# example_2_path = "/blt/graphs/example_2_mar.png"
# plt.savefig(example_2_path)
# plt.close()

# Plot Example 3: MAR y
plt.figure(figsize=(6, 6))
plt.scatter(x[m3 == 0], y[m3 == 0], color="blue", alpha=0.6, label="Observed x")
plt.scatter(x[m3 == 1], y[m3 == 1], color="red", alpha=0.6, label="Missing x")
# plt.title('Example 3: MAR y')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Plot Example 4: MNAR with Shift
plt.figure(figsize=(6, 6))
plt.scatter(z[m4 == 0], x4_obs[m4 == 0], color="blue", alpha=0.6, label="Observed x")
plt.scatter(z[m4 == 1], x4_obs[m4 == 1], color="red", alpha=0.6, label="Missing x")
plt.plot(z, 1 * z, color="black", label="True Projection Line (Non-Missing)")
plt.plot(
    z, 1 * z + shift_value, color="orange", label="Shifted Projection Line (Missing)"
)
# plt.title('Example 4: Shifted Instrument Regression')
plt.xlabel("z_2")
plt.ylabel("x")
plt.legend()
# example_3_path = "/blt/graphs/example_3_mcar_shift.png"
# plt.savefig(example_3_path)
# plt.close()

# Example 5 MCAR with heteroskedasticity
rng = np.random.default_rng(123456)

# Parameters
n_samples = 1000

# Generate z for all examples
z = rng.normal(0, 1, n_samples)
regressors = np.column_stack([np.ones_like(z), np.square(z)])
sigma_xi = np.sqrt(np.dot(regressors, [1, 1]))  # Heteroskedasticity
x = 1 * z + sigma_xi * rng.normal(0, 1, n_samples)  # True x, gamma = 1
m5 = rng.uniform(size=n_samples) < 0.5  # Approximately 50% missing

# Plot Example 5: MCAR with heteroskedasticity
plt.figure(figsize=(6, 6))
plt.scatter(z[m5 == 0], x[m5 == 0], color="blue", alpha=0.6, label="Observed x")
plt.scatter(z[m5 == 1], x[m5 == 1], color="red", alpha=0.6, label="Missing x")
plt.plot(z, 1 * z, color="black", label="True Projection Line")
# plt.title('Example 1 (b): MCAR with heteroskedasticity')
plt.xlabel("z_2")
plt.ylabel("x")
plt.legend()
