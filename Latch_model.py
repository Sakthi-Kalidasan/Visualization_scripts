## Modeling latch time constant from C/ID characterization
import matplotlib.pyplot as plt
import numpy as np

I_base = 269.2e-6
kc_eff = 21.25e-12
kgm_eff = 2.962


def tau_latch(CL, I, kc=kc_eff, kgm=kgm_eff):
    
    Tau_eff = ((kc*I) + CL)/(kgm_eff*I)
    
    return Tau_eff


I_sweep = np.linspace(0.01e-3, 5e-3, 100)
Tau_sweep1 = tau_latch(5e-15, I_sweep)
Tau_sweep2 = tau_latch(20e-15, I_sweep)
Tau_sweep3 = tau_latch(80e-15, I_sweep)
Tau_sweep_ref = tau_latch(0, I_sweep)


Tau_target = 50e-12


plt.figure()
plt.plot(I_sweep*1e3, Tau_sweep1*1e12, label='CL = 5e-15')
plt.plot(I_sweep*1e3, Tau_sweep2*1e12, label='CL = 20e-15')
plt.plot(I_sweep*1e3, Tau_sweep3*1e12, label='CL = 80e-15')
plt.plot(I_sweep*1e3, Tau_sweep_ref*1e12, label='CL = 0')
plt.axhline(y=Tau_target*1e12, color='r', linestyle='--', label='Target Tau = 50 ps')
plt.xlabel('Current (I) [mA]')
plt.ylabel('Tau [ps]')
plt.yscale('log')
plt.title('Tau as a function of Current (I)')
plt.legend()
plt.grid(True)


# Annotate the intersection points of the horizontal line in the plot
for CL, Tau_sweep in zip([5e-15, 20e-15, 80e-15, 0], [Tau_sweep1, Tau_sweep2, Tau_sweep3, Tau_sweep_ref]):
    idx = np.argmin(np.abs(Tau_sweep - Tau_target))
    plt.annotate(f'({I_sweep[idx]*1e3:.2f}, {Tau_sweep[idx]*1e12:.2f})', 
                 (I_sweep[idx]*1e3, Tau_sweep[idx]*1e12), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')


plt.show()


## Noise analysis

Vn_target = 500e-6
KT = 4.11e-21
Gamma = 4/3

CL_target = 2*KT*Gamma/(Vn_target**2)
print(f'CL_target = {CL_target*1e15:.2f} fF')

# Calculate Tau for CL_target
Tau_sweep_target = tau_latch(CL_target, I_sweep)

# Plot Tau for CL_target
plt.figure()
plt.plot(I_sweep*1e3, Tau_sweep_target*1e12, label=f'CL_target = {CL_target*1e15:.2f} fF')
plt.axhline(y=Tau_target*1e12, color='r', linestyle='--', label='Target Tau = 50 ps')
plt.xlabel('Current (I) [mA]')
plt.ylabel('Tau [ps]')
plt.yscale('log')
plt.title('Tau as a function of Current (I) for CL_target')
plt.legend()
plt.grid(True)

# Add text box displaying the noise target
plt.text(0.05, 0.95, f'Noise Target Vn = {Vn_target*1e6:.2f} uVrms', transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Draw a vertical line at the intersection point and annotate it
idx_target = np.argmin(np.abs(Tau_sweep_target - Tau_target))
plt.axvline(x=I_sweep[idx_target]*1e3, color='b', linestyle='--')
plt.annotate(f'I = {I_sweep[idx_target]*1e3:.2f} mA', 
             (I_sweep[idx_target]*1e3, Tau_target*1e12), 
             textcoords="offset points", 
             xytext=(10,-15), 
             ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

plt.show()

Scaling_factor = I_sweep[idx_target]/I_base

print(f'Scaling factor = {Scaling_factor:.2f}')
