import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Function to model the logarithmic fit
def logarithmic_func(x, a, b):
    return a * np.log((0.75*1.8)/(1*x)) + b

# Load the CSV file
data = pd.read_csv(r'new_conv_latch_vin_vs_t_sett_upto_1V_logpoints.csv')

# Assuming the CSV has columns 'x' and 'y'
x_data = data['t_sett_75per_conv X']
y_data = data['t_sett_75per_conv Y']

# Increase the accuracy of the fit by using more iterations and a tighter tolerance
params, covariance = curve_fit(logarithmic_func, x_data, y_data, maxfev=10000, ftol=1e-10)
a, b = params

# Generate y values based on the improved fit
y_fit = logarithmic_func(x_data, a, b)

# Plot the data and the fit
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit, color='red', label='Log fit')

# Display the fitted equation
equation_text = f't_set = {a*1e9:.3f}n * log((0.75*1.8)/(1*vin)) + {b*1e9:.3f}n'
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.legend()
plt.xlabel('vin (V)')
plt.ylabel('settling time (s)')
plt.title('Exponential Fit')
plt.show()