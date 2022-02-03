from itertools import product
from filter import Preprocessor, ECGFilter
import matplotlib.pyplot as plt
import numpy as np


# preprocessing settings
hp_freq = 16
lp_freq = 160
hp_active = True
lp_active = True
subsample_rate = 1

# regular settings
hz = 1000
seconds = 10
samples = int(seconds * hz / subsample_rate)
t = range(samples)
noice_signal_combination = ['data/thorax2.txt']
signal_with_noise = ['data/abdomen3.txt']

# model settings
learning_rate = 0.005
window_size = 45

# postprocessing settings
params = dict()
params['hp_freq'] = [60]
params['lp_freq'] = [120]
params['hp_active'] = [False]
params['lp_active'] = [True]

param_list = []
for values in product(*params.values()):
    param_list.append(dict(zip(params, values)))

# start script

p = Preprocessor(seconds=seconds)
p.set_preprocessing_options(
    hp_freq,
    lp_freq,
    hp_active,
    lp_active,
    subsample_rate
)

signal_w_noise = p.get_signal(signal_with_noise)
noise = p.get_signal(noice_signal_combination)

filter = ECGFilter(signal_w_noise, noise, window_size=window_size, samples=samples)
s_hat = filter.online_stochastic_filter(learning_rate=learning_rate)



for value in param_list:
    print('|')
    p.set_preprocessing_options(
        value['hp_freq'],
        value['lp_freq'],
        value['hp_active'],
        value['lp_active'],
        subsample_rate
    )

    s_hat_filtered = p.filter_signal(s_hat)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(15, 5))

    # Plot the original signal
    ax1.plot(t, noise)
    ax1.set_title('Noise')

    ax2.plot(t, signal_w_noise)
    ax2.set_title('Input + Noise')


    ax3.plot(t, s_hat)
    ax3.set_title('Final result')


    ax4.plot(t, s_hat_filtered)
    ax4.set_title('Final result after postprocessing')

    # plt.savefig(f'test_images/hp_freq={value["hp_freq"]}_lp_freq={value["lp_freq"]}_hp_active={value["hp_active"]}_lp_active={value["lp_active"]}')
    plt.show()
