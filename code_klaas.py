from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import pywt
import statistics

def normalise_signal(signal):
  "Normalise signal by subtracting the mean and dividing by standard deviation"
  signal = signal - statistics.mean(signal)
  signal = signal / np.std(signal)
  return signal 

# Retrieving the signal, consisting of 20 000 sample points measured at 1 kHz
hz = 1000
seconds = 2 # 20
samples = seconds * hz

# todo: merge thorax1 and thorax2 (see third example report)
######### REMEMBER TO GET RID OF THE [:1000]
t = np.linspace(0, seconds, samples, False)
signal_abdomen = np.loadtxt('data/abdomen3.txt')[:samples]

signal_thorax = np.loadtxt('data/thorax1.txt')[:samples]

# Plotting the original thorax1 signal
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(15, 5))

# Plot the original signal
ax1.plot(t, signal_abdomen)
ax1.set_title('Original Abdomen Signal')

# Normalise and plot signal
signal_abdomen = normalise_signal(signal_abdomen)
ax2.plot(t, signal_abdomen)
ax2.set_title('Normalised Abdomen Signal')

# Extract baseline
sos = signal.butter(6, 20, 'lowpass', fs=samples, output='sos')
signal_abdomen_baseline = signal.sosfilt(sos, signal_abdomen)
signal_abdomen_without_baseline_lowpass = signal_abdomen - signal_abdomen_baseline

sos = signal.butter(6, 20, 'highpass', fs=samples, output='sos')
signal_abdomen_without_baseline_highpass = signal.sosfilt(sos, signal_abdomen)

# Plot baseline
ax3.plot(t, signal_abdomen_baseline)
ax3.set_title('Baseline of Abdomen')

ax4.plot(t, signal_abdomen_without_baseline_lowpass)
ax4.set_title('Abdomen without baseline (lowpass)')

ax5.plot(t, signal_abdomen_without_baseline_highpass)
ax5.set_title('Abdomen without baseline (highpass)')

plt.tight_layout()
plt.show()