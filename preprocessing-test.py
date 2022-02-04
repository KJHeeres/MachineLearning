from itertools import product
from filter import *
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

hz = 1000
seconds = 2
subsample_rate = 1
window_size = 250
learning_rate = 0.001
samples = int(seconds * hz / subsample_rate)
t = range(samples)

p = Filter(seconds=seconds)

params = dict()
params['hp_freq'] = [x for x in range(20, 90)]  # niets meer aan doen (was 45)
params['lp_freq'] = [x for x in range(2, 30)]  # niets meer aan doen (was 30)

param_list = []
for values in product(*params.values()):
    param_list.append(dict(zip(params, values)))


for value in param_list:
    # print('|', end='')
    p.set_filter_frequencies(
        value['hp_freq'],
        value['lp_freq'],
    )

    signal_w_noise = p.get_signal(['data/abdomen3.txt'])
    noise = p.get_signal(['data/thorax2.txt'])

    unfiltered_signal_w_noise = p.get_unfiltered_signal(['data/abdomen3.txt'])
    unfiltered_noise = p.get_unfiltered_signal(['data/thorax2.txt'])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(15, 5))

    ax1.plot(t, noise)
    ax1.set_title('Noise')

    ax2.plot(t, signal_w_noise)
    ax2.set_title('Input + Noise')

    ax3.plot(t, noise, label='Noise')
    ax3.plot(t, np.abs(signal.hilbert(noise)), label='Envelope')
    ax3.set_title('Noise with Hilbert transformation')

    ax4.plot(t, signal_w_noise, label='Noise')
    ax4.plot(t, np.abs(signal.hilbert(signal_w_noise)), label='Envelope')
    ax4.set_title('Signal+noise with Hilbert transformation')

    plt.legend()

    plt.savefig(
        f'test_images/hp_freq={value["hp_freq"]}_lp_freq={str(value["lp_freq"]).replace(".","_")}')
    plt.close()
