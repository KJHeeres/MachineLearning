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
thorax_signal = ['data/thorax2.txt']
signal_with_noise = ['data/abdomen3.txt']

# model settings
window_size = 250

# start script

# preprocessing
upper_freq = 90
lower_freq = 27

p = Filter(seconds=seconds)
p.set_filter_frequencies(
    upper_freq,
    lower_freq
)

filtered_abdomen_signal = np.abs(signal.hilbert(p.get_signal(['data/abdomen3.txt'])))
filtered_thorax_signal = np.abs(signal.hilbert(p.get_signal(thorax_signal)))

# windowed linear Regression
m = Model(filtered_abdomen_signal, filtered_thorax_signal, window_size=window_size, samples=samples)
s_hat = m.sliding_window_linear_regressor(is_causal=False)

# postprocessing settings
params = dict()
params['upper_freq'] = [x*10 for x in range(1, 40)]
params['lower_freq'] = [x for x in range(1, 40)]

param_list = []
for values in product(*params.values()):
    param_list.append(dict(zip(params, values)))

for value in param_list:

    if value['upper_freq'] < value['lower_freq']:
        continue

    print('|')
    p.set_filter_frequencies(
        value['upper_freq'],
        value['lower_freq']
    )

    filtered_signal = p.filter_signal(filtered_abdomen_signal - s_hat)
    s_hat_filtered = np.abs(signal.hilbert(filtered_signal))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(15, 5))

    # Plot the original signal
    ax1.plot(t, filtered_thorax_signal)
    ax1.set_title('Filtered Thorax Signal (noise)')

    ax2.plot(t, filtered_abdomen_signal)
    ax2.set_title('Filtered Abdomen Signal (Input + Noise)')

    ax3.plot(t, filtered_abdomen_signal - s_hat)
    ax3.set_title('Abdomen - Thorax (Output)')

    ax4.plot(t, s_hat_filtered)
    ax4.set_title('Filtered Output')

    plt.savefig(f'test_images/upper_freq={value["upper_freq"]}_lower_freq={value["lower_freq"]}')
    plt.close()
