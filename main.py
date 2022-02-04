from itertools import product
from filter import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# regular settings
hz = 1000
seconds = 10
samples = int(seconds * hz)
t = range(samples)
thorax_signal = np.loadtxt('data/thorax2.txt')[:samples]
abdomen_signal = np.loadtxt('data/abdomen3.txt')[:samples]

# model settings
window_size = 250

# start script

# preprocessing
pre_upper_freq = 90
pre_lower_freq = 27

p = Filter(seconds=seconds)
p.set_filter_frequencies(
    pre_upper_freq,
    pre_lower_freq
)

filtered_abdomen_signal = p.get_signal(['data/abdomen3.txt'])
filtered_thorax_signal = p.get_signal(['data/thorax2.txt'])

filtered_abdomen_signal_hilbert = np.abs(signal.hilbert(filtered_abdomen_signal))
filtered_thorax_signal_hilbert = np.abs(signal.hilbert(filtered_thorax_signal))

# windowed linear Regression
m = Model(filtered_abdomen_signal_hilbert, filtered_thorax_signal_hilbert, window_size=window_size, samples=samples)
s_hat = m.sliding_window_linear_regressor(is_causal=False)

# postprocessing settings
params = dict()
post_upper_freq = [180]
post_lower_freq = [25]

print('|')
p.set_filter_frequencies(
    post_upper_freq,
    post_lower_freq
)

filtered_signal = p.filter_signal(filtered_abdomen_signal_hilbert - s_hat)
output = np.abs(signal.hilbert(filtered_signal))

# Originals 
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(15, 6))
fig.tight_layout()

# Plot the original signal
ax1.plot(t, thorax_signal)
ax1.set_title('Original thorax signal')

ax2.plot(t, abdomen_signal)
ax2.set_title('Original abdomen signal')

plt.savefig(f'test_images/final_original_signals')
plt.show()

#Pre-processing figures
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(15, 6))
fig.tight_layout()

ax1.plot(t, filtered_thorax_signal)
ax1.set_title('Pre-processed thorax signal')

ax2.plot(t, filtered_abdomen_signal_hilbert)
ax2.set_title('Pre-processed abdomen signal')

ax3.plot(t, filtered_thorax_signal_hilbert)
ax3.set_title('Pre-processed thorax signal (with hilbert)')

ax4.plot(t, filtered_abdomen_signal_hilbert)
ax4.set_title('Pre-processed abdomen signal (with hilbert)')

plt.savefig(f'test_images/final_preprocessed_signals')
plt.show()

#Processing figures
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(15, 6))
fig.tight_layout()

ax1.plot(t, s_hat)
ax1.set_title('Linear Regression (Output)')

ax2.plot(t, filtered_abdomen_signal_hilbert - s_hat)
ax2.set_title('Abdomen - Thorax (Output)')

plt.savefig(f'test_images/final_processed_signal')
plt.show()

# Post processing figures
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(15, 6))
fig.tight_layout()

ax1.plot(t, thorax_signal)
ax1.set_title('Original thorax signal')

ax2.plot(t, abdomen_signal)
ax2.set_title('Original abdomen signal')

ax3.plot(t, output)
ax3.set_title('Post-processed Output')

plt.savefig(f'test_images/final_post_processed_signal')
plt.show()