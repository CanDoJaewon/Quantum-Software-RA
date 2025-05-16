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

# Truncated Taylor Series functions for cosine and sine with 5, 6, and 7 terms
def truncated_taylor_cos_5(x):
    return 1 - (x**2) / 2 + (x**4) / 24 - (x**6) / math.factorial(6) + (x**8) / math.factorial(8)

def truncated_taylor_sin_5(x):
    return x - (x**3) / 6 + (x**5) / 120 - (x**7) / math.factorial(7) + (x**9) / math.factorial(9)

def truncated_taylor_cos_6(x):
    return 1 - (x**2) / 2 + (x**4) / 24 - (x**6) / math.factorial(6) + (x**8) / math.factorial(8) - (x**10) / math.factorial(10)

def truncated_taylor_sin_6(x):
    return x - (x**3) / 6 + (x**5) / 120 - (x**7) / math.factorial(7) + (x**9) / math.factorial(9) - (x**11) / math.factorial(11)

def truncated_taylor_cos_7(x):
    return 1 - (x**2) / 2 + (x**4) / 24 - (x**6) / math.factorial(6) + (x**8) / math.factorial(8) - (x**10) / math.factorial(10) + (x**12) / math.factorial(12)

def truncated_taylor_sin_7(x):
    return x - (x**3) / 6 + (x**5) / 120 - (x**7) / math.factorial(7) + (x**9) / math.factorial(9) - (x**11) / math.factorial(11) + (x**13) / math.factorial(13)

# Function to calculate squared error
def squared_error(true_values, approx_values):
    return (true_values - approx_values) ** 2

# Range of terms for General Taylor Series (1 to 10)
n_terms_list = range(1, 11)

# Calculate General Taylor Series error for cosine and sine
cos_errors_general = []
sin_errors_general = []

for n_terms in n_terms_list:
    cos_approx_general = np.array([general_taylor_cos(x, n_terms) for x in x_values])
    sin_approx_general = np.array([general_taylor_sin(x, n_terms) for x in x_values])
    cos_errors_general.append(squared_error(true_cos_values, cos_approx_general))
    sin_errors_general.append(squared_error(true_sin_values, sin_approx_general))

# Calculate Truncated Taylor Series error for 5, 6, and 7 terms for cosine and sine
cos_error_truncated_5 = squared_error(true_cos_values, np.array([truncated_taylor_cos_5(x) for x in x_values]))
sin_error_truncated_5 = squared_error(true_sin_values, np.array([truncated_taylor_sin_5(x) for x in x_values]))

cos_error_truncated_6 = squared_error(true_cos_values, np.array([truncated_taylor_cos_6(x) for x in x_values]))
sin_error_truncated_6 = squared_error(true_sin_values, np.array([truncated_taylor_sin_6(x) for x in x_values]))

cos_error_truncated_7 = squared_error(true_cos_values, np.array([truncated_taylor_cos_7(x) for x in x_values]))
sin_error_truncated_7 = squared_error(true_sin_values, np.array([truncated_taylor_sin_7(x) for x in x_values]))

# Plot squared error for General Taylor Series and Truncated Taylor Series (5, 6, and 7 terms) with bold red lines for Truncated Series
plt.figure(figsize=(14, 6))

# Cosine error for General and Truncated Taylor Series
plt.subplot(1, 2, 1)
for n, error in enumerate(cos_errors_general, start=1):
    plt.plot(x_values, error, label=f'General Terms: {n}', linestyle='--', linewidth=0.8)
plt.plot(x_values, cos_error_truncated_5, 'r-', linewidth=2.5, label='Truncated Series (5 terms)')
plt.plot(x_values, cos_error_truncated_6, 'r-', linewidth=2.5, label='Truncated Series (6 terms)')
plt.plot(x_values, cos_error_truncated_7, 'r-', linewidth=2.5, label='Truncated Series (7 terms)')
plt.yscale('log')
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', 'π/2', 'π', '3π/2', '2π'])
plt.xlabel('X values (Between 0 to 2π)')
plt.ylabel('Squared Error (Log Scale)')
plt.title('Cosine Squared Error for General and Truncated Taylor Series')
plt.legend()

# Sine error for General and Truncated Taylor Series
plt.subplot(1, 2, 2)
for n, error in enumerate(sin_errors_general, start=1):
    plt.plot(x_values, error, label=f'General Terms: {n}', linestyle='--', linewidth=0.8)
plt.plot(x_values, sin_error_truncated_5, 'r-', linewidth=2.5, label='Truncated Series (5 terms)')
plt.plot(x_values, sin_error_truncated_6, 'r-', linewidth=2.5, label='Truncated Series (6 terms)')
plt.plot(x_values, sin_error_truncated_7, 'r-', linewidth=2.5, label='Truncated Series (7 terms)')
plt.yscale('log')
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', 'π/2', 'π', '3π/2', '2π'])
plt.xlabel('X values (Between 0 to 2π)')
plt.ylabel('Squared Error (Log Scale)')
plt.title('Sine Squared Error for General and Truncated Taylor Series')
plt.legend()

plt.tight_layout()
plt.show()
