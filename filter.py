from scipy import signal

import numpy as np
import pywt
import statistics
from sklearn.linear_model import LinearRegression, SGDRegressor
from numpy.lib.stride_tricks import sliding_window_view

class Preprocessor:

    def __init__(self, seconds=10):
        self.cutoff_freq_hp = 12
        self.cutoff_freq_lp = 120
        self.highpass_active = True
        self.lowpass_acitve = True
        self.subsample_rate = 1
        self.samples = 1000 * seconds

    def set_preprocessing_options(self, freq_hp, freq_lp, hp_active, lp_active, subsample_rate):
        self.cutoff_freq_hp = freq_hp
        self.cutoff_freq_lp = freq_lp
        self.highpass_active = hp_active
        self.lowpass_acitve = lp_active
        self.subsample_rate = subsample_rate

    def filter_signal(self, signal):
        s = self.normalise_signal(signal)
        s = self.apply_smoothening_filters(s)
        return self.subsample_signal(s)

    def get_signal(self, signal_file_names):
        s = self.load_signal(signal_file_names)
        return self.filter_signal(s)

    def load_signal(self, signals):
        signal_out = np.zeros(self.samples)
        for signal in signals:
            signal_out += np.loadtxt(signal)[:self.samples]

        return signal_out / len(signals)

    def apply_smoothening_filters(self, input_signal):
        result = input_signal
        if self.highpass_active:
            sos = signal.butter(6, self.cutoff_freq_hp, 'highpass', fs=self.samples, output='sos')
            result = signal.sosfilt(sos, result)
        if self.lowpass_acitve:
            sos = signal.butter(6, self.cutoff_freq_lp, 'lowpass', fs=self.samples, output='sos')
            result = result - signal.sosfilt(sos, result)

        return result

    def normalise_signal(self, signal):
        "Normalise signal by subtracting the mean and dividing by standard deviation"
        signal = signal - statistics.mean(signal)
        signal = signal / np.std(signal)
        return signal

    def subsample_signal(self, signal):
        return signal[::self.subsample_rate]

class ECGFilter:

    def __init__(self, signal, noise, window_size=500, samples=10000):
        self.samples = samples
        self.window_size = window_size
        self.s = signal
        self.n = noise

    def create_windows(self, is_causal):
        if is_causal:
            n_windows = sliding_window_view(self.n, self.window_size + 1)
            s_targets = self.s[self.window_size:self.samples]
        else:
            n_windows = sliding_window_view(self.n, 2 * self.window_size + 1)
            s_targets = self.s[self.window_size:(self.samples- self.window_size)]

        return n_windows, s_targets

    def shift_signal(self, signal, is_causal):
        if is_causal:
            signal = np.append(signal, np.zeros(self.window_size))
            return np.roll(signal, self.window_size)

        signal = np.append(signal, np.zeros(2 * self.window_size))
        return np.roll(signal, self.window_size)

    def sliding_window_linear_regressor(self, is_causal=True):
        model = LinearRegression()
        samples, targets = self.create_windows(is_causal)
        model.fit(samples, targets)

        y = model.predict(samples)

        return self.shift_signal(y, is_causal)

    def sliding_window_stochastic_regressor(self, is_causal=True):
        model = SGDRegressor()
        samples, targets = self.create_windows(is_causal)
        model.fit(samples, targets)

        y = model.predict(samples)

        return self.shift_signal(y, is_causal)

    def online_stochastic_filter(self, learning_rate=0.005):
        model = SGDRegressor(learning_rate='constant', eta0=learning_rate)
        s_hat = np.array([])

        sample = self.n[0:(self.window_size)]
        sample = np.reshape(sample, (1, -1))

        model.partial_fit(sample, [self.s[self.window_size] ])
        for idx in range(self.window_size, self.samples):
            sample = self.n[(idx - self.window_size):idx]
            sample = np.reshape(sample, (1, -1))

            y = model.predict(sample)
            cur_s_hat = self.s[idx] - y

            model.partial_fit(sample, np.array([self.s[idx]]))

            s_hat = np.append(s_hat, cur_s_hat)

        return self.shift_signal(s_hat, True)
