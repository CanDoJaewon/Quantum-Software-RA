# Updated code to generate the graph with squared error on log scale for better detail on accuracy changes

import numpy as np
import matplotlib.pyplot as plt
import math

# Define exact sine and cosine values using numpy for comparison
x_values = np.linspace(0, 2 * np.pi, 100)
true_cos_values = np.cos(x_values)
true_sin_values = np.sin(x_values)

# General Taylor Series function for cosine and sine
def general_taylor_cos(x, n_terms):
    approx = 0
    for n in range(n_terms):
        approx += ((-1) ** n * x ** (2 * n)) / math.factorial(2 * n)
    return approx

def general_taylor_sin(x, n_terms):
    approx = 0
    for n in range(n_terms):
        approx += ((-1) ** n * x ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return approx

# Truncated Taylor Series for cosine and sine (limited number of terms)
def truncated_taylor_cos(x):
    return 1 - (x**2) / 2 + (x**4) / 24

def truncated_taylor_sin(x):
    return x - (x**3) / 6 + (x**5) / 120

# Function to calculate squared error
def squared_error(true_values, approx_values):
    return (true_values - approx_values) ** 2

# Range of terms for general Taylor Series (1 to 10)
n_terms_list = range(1, 11)

# Calculate general Taylor Series error for cosine and sine
cos_errors_general = []
sin_errors_general = []

for n_terms in n_terms_list:
    cos_approx_general = np.array([general_taylor_cos(x, n_terms) for x in x_values])
    sin_approx_general = np.array([general_taylor_sin(x, n_terms) for x in x_values])
    cos_error_general = squared_error(true_cos_values, cos_approx_general)
    sin_error_general = squared_error(true_sin_values, sin_approx_general)
    cos_errors_general.append(cos_error_general)
    sin_errors_general.append(sin_error_general)

# Calculate truncated Taylor Series error for cosine and sine
cos_approx_truncated = np.array([truncated_taylor_cos(x) for x in x_values])
sin_approx_truncated = np.array([truncated_taylor_sin(x) for x in x_values])
cos_error_truncated = squared_error(true_cos_values, cos_approx_truncated)
sin_error_truncated = squared_error(true_sin_values, sin_approx_truncated)

# Plot the results on log scale for better visualization of small errors
plt.figure(figsize=(14, 6))

# Cosine plot: General Taylor Series and Truncated Taylor Series
plt.subplot(1, 2, 1)
for n, error in enumerate(cos_errors_general, start=1):
    plt.plot(x_values, error, label=f'Terms: {n}')
plt.plot(x_values, cos_error_truncated, 'k--', label='Truncated Series')
plt.yscale('log')
plt.xlabel('X values (Radians)')
plt.ylabel('Squared Error (Log Scale)')
plt.title('Cosine Squared Error for General and Truncated Taylor Series')
plt.legend()

# Sine plot: General Taylor Series and Truncated Taylor Series
plt.subplot(1, 2, 2)
for n, error in enumerate(sin_errors_general, start=1):
    plt.plot(x_values, error, label=f'Terms: {n}')
plt.plot(x_values, sin_error_truncated, 'k--', label='Truncated Series')
plt.yscale('log')
plt.xlabel('X values (Radians)')
plt.ylabel('Squared Error (Log Scale)')
plt.title('Sine Squared Error for General and Truncated Taylor Series')
plt.legend()

plt.tight_layout()
plt.show()
