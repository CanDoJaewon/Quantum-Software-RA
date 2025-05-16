import numpy as np
import matplotlib.pyplot as plt
import math

# Step 1: Generate an array of angles from 0 to 2π
x_values = np.linspace(0, 2 * np.pi, 100)

# Step 2: Create arrays with the exact cosine and sine values
true_cos_values = np.cos(x_values)
true_sin_values = np.sin(x_values)

# Define truncated Taylor series (as given in the document)
def truncated_taylor_cos(x):
    """Truncated Taylor series approximation of cosine up to x^4."""
    return 1 - (x**2) / 2 + (x**4) / 24

def truncated_taylor_sin(x):
    """Truncated Taylor series approximation of sine up to x^5."""
    return x - (x**3) / 6 + (x**5) / 120

# Define general Taylor series approximation
def general_taylor_cos(x, n_terms):
    """General Taylor series approximation of cosine with n terms."""
    approx = 0
    for n in range(n_terms):
        approx += ((-1) ** n * x ** (2 * n)) / math.factorial(2 * n)
    return approx

def general_taylor_sin(x, n_terms):
    """General Taylor series approximation of sine with n terms."""
    approx = 0
    for n in range(n_terms):
        approx += ((-1) ** n * x ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return approx

# Function to calculate mean squared error (MSE)
def mean_squared_error(true_values, approx_values):
    return np.mean((true_values - approx_values) ** 2)

# Step 3: Compare the truncated Taylor series with the exact values
truncated_cos_approx = [truncated_taylor_cos(x) for x in x_values]
truncated_sin_approx = [truncated_taylor_sin(x) for x in x_values]

# Calculate errors for truncated series
truncated_cos_error = mean_squared_error(true_cos_values, truncated_cos_approx)
truncated_sin_error = mean_squared_error(true_sin_values, truncated_sin_approx)

print(f"Truncated Cosine Error: {truncated_cos_error}")
print(f"Truncated Sine Error: {truncated_sin_error}")

# Step 4: Plot errors as a function of number of terms for general Taylor series
n_terms_list = range(1, 11)  # Test with 1 to 10 terms
cos_errors = []
sin_errors = []

for n_terms in n_terms_list:
    cos_approx = [general_taylor_cos(x, n_terms) for x in x_values]
    sin_approx = [general_taylor_sin(x, n_terms) for x in x_values]
    
    cos_errors.append(mean_squared_error(true_cos_values, cos_approx))
    sin_errors.append(mean_squared_error(true_sin_values, sin_approx))

# Step 5: Plot the results
plt.figure(figsize=(12, 6))

# Plot for cosine
plt.subplot(1, 2, 1)
plt.plot(n_terms_list, cos_errors, marker='o', label='Cosine Error')
plt.xlabel('Number of Terms in Taylor Series')
plt.ylabel('Mean Squared Error')
plt.title('Error of Cosine Approximation')
plt.grid(True)

# Plot for sine
plt.subplot(1, 2, 2)
plt.plot(n_terms_list, sin_errors, marker='o', label='Sine Error', color='orange')
plt.xlabel('Number of Terms in Taylor Series')
plt.ylabel('Mean Squared Error')
plt.title('Error of Sine Approximation')
plt.grid(True)

plt.tight_layout()
plt.show()
