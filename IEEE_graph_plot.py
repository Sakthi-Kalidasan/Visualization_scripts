import numpy as np

import matplotlib.pyplot as plt

# Define a function to set the style for IEEE conference
def set_ieee_style():
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times'],
    })

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Apply IEEE style
set_ieee_style()

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y, label='Sine Wave')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Example Plot for IEEE Conference')
ax.legend()

# Save the plot as a high-resolution PNG file
plt.savefig('ieee_plot.png', bbox_inches='tight')

# Show the plot
plt.show()