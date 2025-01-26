import numpy as np

# Constants
Vsupply = 1.8
kT = 4.11e-21
Gamma = 2/3

# Performance Targets
Vout = 0.9
PSRR_dB = -40
LG_dB = -1 * PSRR_dB
LG = 10 ** (LG_dB / 20)

LDO_bandwidth = 1e6
LDO_Noise = 100e-6

I_load = 1e-3
C_load = 100e-12

# Initial Assumptions
Vref = 0.8

# Feedback resistors consume about 5 % of load current
I_res = 0.05 * I_load
Rf_total = Vout / I_res

Rf_2 = (Vref / Vout) * Rf_total
Rf_1 = Rf_total - Rf_2

# Bandwidth target for Amp is 10 times the LDO target
Amp_bandwidth = 10 * LDO_bandwidth

# CoverID optimization
# Estimating gm required to meet the LDO bandwidth
I_pass = I_load + I_res

# Assumptions for testing
Rout_eff = 1 / (2 * np.pi * LDO_bandwidth * C_load)
ro = 1 / ((1 / Rout_eff) - (1 / Rf_total))
gm_gds = 25
kcgs = 187.4e-12

Cgs_pass = kcgs * I_pass
gm_pass = gm_gds / ro
kgm_pass = gm_pass / I_pass

Av_amp = LG / ((Rf_2 / Rf_total) * gm_pass * Rout_eff)
Av_amp_dB = 20 * np.log10(Av_amp)

GBW_amp = Av_amp * Amp_bandwidth
GBW_pass = gm_pass * Rout_eff * LDO_bandwidth

C_load_amp = Cgs_pass * (gm_pass * Rout_eff + 1)

# Noise estimations
Vn2_LDO = (LDO_Noise ** 2) / LDO_bandwidth

Factor1 = (1 + (Rf_1 / Rf_2)) ** 2

Vn2_pass = (4 * kT * Gamma * (1 / gm_pass)) / (Av_amp ** 2)
Vn2_fb = 4 * kT * ((Rf_2 * Rf_1) / Rf_total)
Vn2_amp = (Vn2_LDO / Factor1) - Vn2_pass - Vn2_fb

Amp_noise = np.sqrt(Vn2_amp * Amp_bandwidth)

# Displaying amplifier noise and gain bandwidth
GBW_amp_GHz = GBW_amp * 1e-9
Amp_noise_uVrms = Amp_noise * 1e6

print(f'The amplifier gain bandwidth is {GBW_amp_GHz:.2f} GHz')
print(f'The amplifier noise is {Amp_noise_uVrms:.2f} uVrms')
