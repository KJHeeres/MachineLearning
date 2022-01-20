from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import pywt
import statistics

def normalise_signal(signal):
  "Normalise signal to a domain of [-1, 1]"
  max = np.amax(np.absolute(signal))
  return signal / max

def shift_signal(signal, shift):
  signal = np.append(signal, np.zeros(shift))
  return np.roll(signal, shift)

# Retrieving the signal, consisting of 20 000 sample points measured at 1
hz = 1000
seconds = 2
samples = seconds * hz

# todo: merge thorax1 and thorax2 (see third example report)
######### REMEMBER TO GET RID OF THE [:samples]
t = np.linspace(0, seconds, samples, False)
t = range(samples)

signal_abdomen = np.loadtxt('data/abdomen3.txt')[:samples]
signal_thorax = np.loadtxt('data/thorax2.txt')[:samples]

# Apply the highpass filter
sos = signal.butter(6, 20, 'highpass', fs=samples, output='sos')
signal_abdomen_without_baseline_highpass = signal.sosfilt(sos, signal_abdomen)
signal_thorax_without_baseline_highpass = signal.sosfilt(sos, signal_thorax)

# correlate abdomen with thorax
t_correlation = range(2 * samples - 1)
correlation = signal.correlate(signal_abdomen_without_baseline_highpass, signal_thorax_without_baseline_highpass)

print("Index of max correlation value")
shift = np.argmax(correlation) - samples
print(shift)

shifted_thorax = shift_signal(signal_thorax_without_baseline_highpass, shift)
new_abdomen = np.append(signal_abdomen_without_baseline_highpass, np.zeros(shift))
sub = normalise_signal(new_abdomen) - normalise_signal(shifted_thorax)

sub_high_pass = signal.sosfilt(sos, sub)

# Plotting the original thorax1 signal
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(15, 5))

# Plot the original signal
ax1.plot(t, signal_abdomen_without_baseline_highpass)
ax1.set_title('Abdomen without baseline (highpass)')

ax2.plot(t, signal_thorax_without_baseline_highpass)
ax2.set_title('Thorax without baseline (highpass)')

t = range(samples + shift)

ax3.plot(t_correlation, correlation)
ax3.set_title('Abdomen Thorax correlation')

ax4.plot(t, sub)
ax4.plot(t, sub_high_pass)
ax4.set_title('Abdomen - Thorax')

ax5.plot(t, normalise_signal(new_abdomen))
ax5.plot(t, normalise_signal(shifted_thorax))
ax5.plot(t, sub)
ax5.legend(['shifted_abdomen', 'shifted_thorax', 'abdomen - thorax'])

plt.tight_layout()
plt.show()

plt.figure(2)
plt.plot(t, normalise_signal(new_abdomen))
plt.plot(t, normalise_signal(shifted_thorax))
plt.legend(['new_abdomen', 'shifted_thorax'])

plt.show()
