#!/usr/bin/env python3

from scipy import signal

import numpy as np
import matplotlib.pyplot as plt


def demo():
    # Signal generation: 10Hz and 20Hz, sampled at 1 kHz
    t = np.linspace(0, 1, 1000, False)
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)

    # Plotting the original signal
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, sig)
    ax1.set_title('Original Signal')
    ax1.axis([0, 1, -2, 2])

    # Processing the signal by applying a high-pass filter of 15Hz
    sos = signal.butter(10, 15, 'highpass', fs=1000, output='sos')
    sigf = signal.sosfilt(sos, sig)

    # Plotting the processed signal
    ax2.plot(t, sigf)
    ax2.set_title('Processed Signal')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time in seconds')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo()
