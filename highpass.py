#!/usr/bin/env python3

# Useful URLs
# - https://stackoverflow.com/questions/63320705/what-are-order-and-critical-frequency-when-creating-a-low-pass-filter-using
# - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

from scipy import signal

import numpy as np
import matplotlib.pyplot as plt


def run():
    # Retrieving the signal, consisting of 20 000 sample points measured at 1 kHz
    hz = 1000
    seconds = 20
    samples = seconds * hz

    # todo: merge thorax1 and thorax2 (see third example report)
    t = np.linspace(0, seconds, samples, False)
    sig_thorax = np.loadtxt('data/thorax1.txt')
    sig_abdomen = np.loadtxt('data/abdomen3.txt')

    # Retrieving the frequencies (should perhaps be fftfreq or fft/rfft?). Do we even need to do this at all?
    # frequencies = np.fft.rfftfreq(samples, 1 / hz)
    # print(np.sort(frequencies))

    # Plotting the original signal
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, sharex=True, sharey=True, figsize=(15, 5))
    ax1.plot(t, sig_thorax)
    ax1.set_title('Original Thorax Signal')

    # Attempt to extract the mother's heartbeat
    sos = signal.butter(10, 3, 'highpass', fs=samples, output='sos')
    sigf = signal.sosfilt(sos, sig_thorax)

    ax2.plot(t, sigf)
    ax2.set_title('Processed Signal')

    ax3.plot(t, sig_abdomen)
    ax3.set_title('Original Abdomen Signal')

    ax4.plot(t, sig_abdomen - sigf)
    ax4.set_title('Abdomen without Mother\'s Heartbeat')
    ax4.set_xlabel('Time in seconds')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
