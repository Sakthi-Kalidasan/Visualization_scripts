
## Script to create a plot with IEEE style and save it as a high-resolution PNG file

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl
import csv
from scipy.stats import norm
print(mpl.rcParams['text.latex.preamble'])
print(mpl.rcParams['pgf.texsystem'])

# Define a function to set the style for IEEE conference
def set_ieee_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (4, 3.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times'],
    })

# Example data
T = [0,27,105]
ss_ck2q = [1.85e-10,1.7e-10,1.45e-10]
ss_ck2q_ps = [x*1e12 for x in ss_ck2q]

sf_ck2q = [1.54e-10,1.43e-10,1.26e-10]
sf_ck2q_ps = [x*1e12 for x in sf_ck2q]

tt_ck2q = [1.4e-10,1.32e-10,1.26e-10]
tt_ck2q_ps = [x*1e12 for x in tt_ck2q]

fs_ck2q = [1.27e-10,1.21e-10,1.1e-10]
fs_ck2q_ps = [x*1e12 for x in fs_ck2q]

ff_ck2q = [1.07e-10,1.03e-10,9.72e-11]
ff_ck2q_ps = [x*1e12 for x in ff_ck2q]

# Apply IEEE style
set_ieee_style()

# Create the plot
fig, ax = plt.subplots()
ax.plot(T, ss_ck2q_ps, label='ss corner')
ax.plot(T, sf_ck2q_ps, label='sf corner')
ax.plot(T, tt_ck2q_ps, label='tt corner')
ax.plot(T, fs_ck2q_ps, label='fs corner')
ax.plot(T, ff_ck2q_ps, label='ff corner')
# Add labels and title
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Ck2Q (ps) ')
ax.set_title('PVT Impact on Ck2Q')
ax.legend()

# Save the plot as a high-resolution PNG file
plt.savefig('pvt_ck2q.png', bbox_inches='tight')

# Show the plot
plt.show()

## Next Figure

T = [0,27,105]
ss_noise = [9.5e-5,1.07e-4,1.43e-4]
ss_noise_uV = [x*1e6 for x in ss_noise]

sf_noise = [1.04e-4,1.15e-4,1.51e-4]
sf_noise_uV = [x*1e6 for x in sf_noise]

tt_noise = [1.04e-4,1.14e-4,1.52e-4]
tt_noise_uV = [x*1e6 for x in tt_noise]

fs_noise = [1.05e-4,1.17e-4,1.53e-4]
fs_noise_uV = [x*1e6 for x in fs_noise]

ff_noise = [1.12e-4,1.22e-4,1.59e-4]
ff_noise_uV = [x*1e6 for x in ff_noise]

# Apply IEEE style
set_ieee_style()

# Create the plot
fig, ax = plt.subplots()
ax.plot(T, ss_noise_uV, label='ss corner')
ax.plot(T, sf_noise_uV, label='sf corner')
ax.plot(T, tt_noise_uV, label='tt corner')
ax.plot(T, fs_noise_uV, label='fs corner')
ax.plot(T, ff_noise_uV, label='ff corner')
# Add labels and title
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Noise (uVrms) ')
ax.set_title('PVT Impact on noise')
ax.legend()

# Save the plot as a high-resolution PNG file
plt.savefig('pvt_noise.png', bbox_inches='tight')

# Show the plot
plt.show()


## PVT Speed Calibration plot

ss_ck2q_calib_ps = [278,320]
sf_ck2q_calib_ps = [307,315.5]
tt_ck2q_calib_ps = [291,303]
fs_ck2q_calib_ps = [284.5,289.5]
ff_ck2q_calib_ps = [300,313]

# Apply IEEE style
set_ieee_style()

#Corners used
corner = ['ss','sf','tt','fs','ff']
ck2q_calib_ps_min = [278,307,291,284.5,300]
ck2q_calib_ps_max = [320,315.5,303,289.5,313]

# Create the plot
fig, ax = plt.subplots()

ax.scatter(corner, ck2q_calib_ps_min, label='0°C')
ax.scatter(corner, ck2q_calib_ps_max, label='105°C')

# Draw vertical lines connecting the min and max values
for i in range(len(corner)):
    ax.plot([corner[i], corner[i]], [ck2q_calib_ps_min[i], ck2q_calib_ps_max[i]], color='gray', linestyle='--')

# Add labels and title
ax.set_xlabel('Corners')
ax.set_ylabel('Ck2q (ps)')
ax.set_title('Calibrated PVT Impact on speed')
ax.legend()

# Save the plot as a high-resolution PNG file
plt.savefig('pvt_calib_speed.png', bbox_inches='tight')

# Show the plot
plt.show()

## PVT Noise Calibration plot

# Apply IEEE style
set_ieee_style()

#Corners used
corner = ['ss','sf','tt','fs','ff']
noise_calib_uV_min = [53.26,51.66,49.37,49.14,48.26]
noise_calib_uV_max = [72.17,66.9,65.56,62.2,59.5]

# Create the plot
fig, ax = plt.subplots()

ax.scatter(corner, noise_calib_uV_min, label='0°C')
ax.scatter(corner, noise_calib_uV_max, label='105°C')

# Draw vertical lines connecting the min and max values
for i in range(len(corner)):
    ax.plot([corner[i], corner[i]], [noise_calib_uV_min[i], noise_calib_uV_max[i]], color='gray', linestyle='--')

# Add labels and title
ax.set_xlabel('Corners')
ax.set_ylabel('Noise (uVrms)')
ax.set_title('Calibrated PVT Impact on noise')
ax.legend()

# Save the plot as a high-resolution PNG file
plt.savefig('pvt_calib_noise.png', bbox_inches='tight')

# Show the plot
plt.show()

## Supply voltage vs Load cap comparison plot

# Apply IEEE style
set_ieee_style()

#Corners used
corner = ['ss','sf','tt','fs','ff']
ck2q_load_cap_ps = [400,396.4,376.5,345,352]
ck2q_supply_v = [363.4,357,334,326,327]

# Create the plot
fig, ax = plt.subplots()

ax.scatter(corner, ck2q_load_cap_ps, label='Load cap')
ax.scatter(corner, ck2q_supply_v, label='Supply voltage')

# Add labels and title
ax.set_xlabel('Corners')
ax.set_ylabel('Ck2q (ps)')
ax.set_title('Supply voltage vs Load cap comparison')
ax.legend()

# Save the plot as a high-resolution PNG file
plt.savefig('ck2q_supply_v_load_comp.png', bbox_inches='tight')

# Show the plot
plt.show()




## Monte carlo simulation plot for offset

# Apply IEEE style
set_ieee_style()

# Read data from CSV file
monte_carlo_data = []
with open('Offset_comparator.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        monte_carlo_data.append([float(value) for value in row])

# Convert to numpy array for easier manipulation
monte_carlo_data = np.array(monte_carlo_data)

# Extract x and y values
x_values = monte_carlo_data[:, 0]
y_values = monte_carlo_data[:, 1]

# Create the plot
fig, ax = plt.subplots()

ax.scatter(x_values, y_values, label='Input Offset', alpha=0.6)

# Add labels and title
ax.set_xlabel('Input Offset (mV)')
ax.set_ylabel('frequency')
#ax.set_title('Monte Carlo Simulation Results')
ax.legend()

# Fit a normal distribution to the data
(mu, sigma) = norm.fit(x_values)

# Compute the maximum y value
max_y = np.max(y_values)

# Compute the average y value
avg_y = np.mean(y_values)

# Compute the scaling factor
#scaling_factor = max_y / avg_y
scaling_factor = 6.7

# Plot the PDF of the fitted normal distribution
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
# Draw a vertical line between the p plot and the x-axis
ax.vlines(x, 0, p/scaling_factor, colors='b', linestyles='dotted', linewidth=0.5)

# Add a vertical red line at the mean
ax.vlines(mu, 0, norm.pdf(mu, mu, sigma) / scaling_factor, colors='r', linestyles='dotted', linewidth=1)

# Add vertical lines at multiple sigma intervals
for i in range(1, 4):  # Adding lines for 1, 2, and 3 sigma intervals
    ax.vlines(mu + i * sigma, 0, norm.pdf(mu + i * sigma, mu, sigma) / scaling_factor, colors='r', linestyles='dotted', linewidth=1)
    ax.vlines(mu - i * sigma, 0, norm.pdf(mu - i * sigma, mu, sigma) / scaling_factor, colors='r', linestyles='dotted', linewidth=1)
# Add sigma value to the plot near the plot
# Add sigma value to the plot near the plot
# Add sigma value to the plot near the plot
ax.text(mu + sigma, norm.pdf(mu + sigma, mu, sigma) / scaling_factor + 0.02 * max_y, 
    f'$\sigma = {sigma*1e3:.2f}mV$', fontsize=10, verticalalignment='bottom', 
    horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Add sigma value to the plot on the other side of the mean
ax.text(mu - sigma, norm.pdf(mu - sigma, mu, sigma) / scaling_factor + 0.02 * max_y, 
    f'$\sigma = {sigma*1e3:.2f}mV$', fontsize=10, verticalalignment='bottom', 
    horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Add legend
ax.legend()

# Save the plot as a high-resolution PNG file
plt.savefig('monte_carlo_offset.png', bbox_inches='tight')

# Show the plot
plt.show()
