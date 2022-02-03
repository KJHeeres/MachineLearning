from itertools import product
from filter import Preprocessor, ECGFilter
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

hp_freq = 90
lp_freq = 27
hp_active = False
lp_active = False
subsample_rate = 1

hz = 1000
seconds = 6
learning_rate = 0.001
samples = int(seconds * hz / subsample_rate)
t = range(samples)

p = Preprocessor(seconds=seconds)
p.set_preprocessing_options(
    hp_freq,
    lp_freq,
    hp_active,
    lp_active,
    subsample_rate
)

noice_signal_combinations = [
    ['data/thorax2.txt']
]

params = dict()
params['is_linear'] = [True, False]
params['is_causal'] = [True, False]
params['window_size'] = [200,250,300,350]

param_list = []
for values in product(*params.values()):
    param_list.append(dict(zip(params, values)))


signal_w_noise = np.abs(signal.hilbert(p.get_signal(['data/abdomen3.txt'])))

for noice_signal_combination in noice_signal_combinations:
    noise = np.abs(signal.hilbert(p.get_signal(noice_signal_combination)))
    for value in param_list:
        print('|', end='')

        filter = ECGFilter(signal_w_noise, noise, window_size=value["window_size"], samples=samples)
        if(value["is_linear"]):
            s_hat = filter.sliding_window_linear_regressor(is_causal=value["is_causal"])
        else:
            s_hat = filter.sliding_window_stochastic_regressor(is_causal=value["is_causal"])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(15, 5))

        # Plot the original signal
        ax1.plot(t, noise)
        ax1.set_title('Noise')

        ax2.plot(t, signal_w_noise)
        ax2.set_title('Input + Noise')

        ax3.plot(t, signal_w_noise - s_hat)
        ax3.set_title('Final result')

        plt.savefig(
            f'test_images/noice_signal_combination={noice_signal_combination[0][5:-4]}{len(noice_signal_combination)}_is_linear={value["is_linear"]}_is_causal={value["is_causal"]}_window_size={value["window_size"]}')

params = dict()
params['window_size'] = [45, 125, 250, 500, 1000, 1500]
params['learning_rate'] = [0.001, 0.0005, 0.0001]

param_list = []
for values in product(*params.values()):
    param_list.append(dict(zip(params, values)))
"""
for noice_signal_combination in noice_signal_combinations:
    noise = p.get_signal(noice_signal_combination)
    for value in param_list:
        print('|| ', end='')

        filter = ECGFilter(signal_w_noise, noise, window_size=value["window_size"], samples=samples)

        s_hat = filter.online_stochastic_filter(learning_rate=value["learning_rate"])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(15, 5))

        # Plot the original signal
        ax1.plot(t, noise)
        ax1.set_title('Noise')

        ax2.plot(t, signal_w_noise)
        ax2.set_title('Input + Noise')

        ax3.plot(t, s_hat)
        ax3.set_title('Final result')

        # plt.show()
        plt.savefig(
            f'test_images/noice_signal_combination={noice_signal_combination[0][5:-4]}{len(noice_signal_combination)}_learning_rate={str(value["learning_rate"])[2:]}_window_size={value["window_size"]}')
"""
