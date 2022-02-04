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

filtered_abdomen_signal = np.abs(signal.hilbert(p.get_signal(['data/abdomen3.txt'])))
filtered_thorax_signal = np.abs(signal.hilbert(p.get_signal(['data/thorax2.txt'])))

# windowed linear Regression
m = Model(filtered_abdomen_signal, filtered_thorax_signal, window_size=window_size, samples=samples)
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

filtered_signal = p.filter_signal(filtered_abdomen_signal - s_hat)
output = np.abs(signal.hilbert(filtered_signal))

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(15, 6))

# Plot the original signal
ax1.plot(t, thorax_signal)
ax1.set_title('thorax signal')

ax2.plot(t, abdomen_signal)
ax2.set_title('abdomen_signal')

ax3.plot(t, filtered_thorax_signal)
ax3.set_title('Filtered Thorax Signal (noise)')

ax4.plot(t, filtered_abdomen_signal)
ax4.set_title('Filtered Abdomen Signal (Input + Noise)')

ax5.plot(t, filtered_abdomen_signal - s_hat)
ax5.set_title('Abdomen - Thorax (Output)')

ax6.plot(t, output)
ax6.set_title('Filtered Output')

plt.savefig(f'test_images/final')
plt.show()