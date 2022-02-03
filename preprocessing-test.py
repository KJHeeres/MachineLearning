from itertools import product
from filter import Preprocessor
import matplotlib.pyplot as plt

hz = 1000
seconds = 2
subsample_rate = 1
window_size = 250
learning_rate = 0.001
samples = int(seconds * hz / subsample_rate)
t = range(samples)

p = Preprocessor(seconds=seconds)

params = dict()
params['hp_freq'] = [140]
params['lp_freq'] = [60]
params['hp_active'] = [False]
params['lp_active'] = [False]

param_list = []
for values in product(*params.values()):
    param_list.append(dict(zip(params, values)))


for value in param_list:
    print('|')
    p.set_preprocessing_options(
        value['hp_freq'],
        value['lp_freq'],
        value['hp_active'],
        value['lp_active'],
        subsample_rate
    )

    signal_w_noise = p.get_signal(['data/abdomen3.txt'])
    noise = p.get_signal(['data/thorax2.txt'])

    unfiltered_signal_w_noise = p.get_unfiltered_signal(['data/abdomen3.txt'])
    unfiltered_noise = p.get_unfiltered_signal(['data/thorax2.txt'])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(15, 5))

    ax1.plot(t, unfiltered_noise)
    ax1.set_title('Unfiltered Noise')

    ax2.plot(t, unfiltered_signal_w_noise)
    ax2.set_title('Unfiltered Input + Noise')

    ax3.plot(t, noise)
    ax3.set_title('Noise')

    ax4.plot(t, signal_w_noise)
    ax4.set_title('Input + Noise')

    plt.savefig(f'test_images/hp_freq={value["hp_freq"]}_lp_freq={str(value["lp_freq"]).replace(".","_")}_hp_active={value["hp_active"]}_lp_active={value["lp_active"]}')
