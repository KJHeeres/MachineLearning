from itertools import product
from filter import Preprocessor
import matplotlib.pyplot as plt

hz = 1000
seconds = 6
subsample_rate = 1
window_size = 250
learning_rate = 0.001
samples = int(seconds * hz / subsample_rate)
t = range(samples)

p = Preprocessor(seconds=seconds)

params = dict()
params['hp_freq'] = [12, 16, 20, 24, 36]
params['lp_freq'] = [80, 120, 160, 200]
params['hp_active'] = [True, False]
params['lp_active'] = [True, False]

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
    noise = p.get_signal(['data/thorax1.txt'])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(15, 5))

    ax1.plot(t, noise)
    ax1.set_title('Noise')

    ax2.plot(t, signal_w_noise)
    ax2.set_title('Input + Noise')

    plt.savefig(f'test_images/hp_freq={value["hp_freq"]}_lp_freq={value["lp_freq"]}_hp_active={value["hp_active"]}_lp_active={value["lp_active"]}')
