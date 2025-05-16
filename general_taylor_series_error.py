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

# Plot the results on log scale for better visualization of small errors
plt.figure(figsize=(14, 6))

# Cosine plot for General Taylor Series
plt.subplot(1, 2, 1)
for n, error in enumerate(cos_errors_general, start=1):
    plt.plot(x_values, error, label=f'Terms: {n}')
plt.yscale('log')
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', 'π/2', 'π', '3π/2', '2π'])
plt.xlabel('X values (Between 0 to 2π)')
plt.ylabel('Squared Error (Log Scale)')
plt.title('Cosine Squared Error for General Taylor Series')
plt.legend()

# Sine plot for General Taylor Series
plt.subplot(1, 2, 2)
for n, error in enumerate(sin_errors_general, start=1):
    plt.plot(x_values, error, label=f'Terms: {n}')
plt.yscale('log')
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', 'π/2', 'π', '3π/2', '2π'])
plt.xlabel('X values (Between 0 to 2π)')
plt.ylabel('Squared Error (Log Scale)')
plt.title('Sine Squared Error for General Taylor Series')
plt.legend()

plt.tight_layout()
plt.show()
