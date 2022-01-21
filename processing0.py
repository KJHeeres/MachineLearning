from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import pywt
import statistics
from sklearn.linear_model import LinearRegression
from numpy.lib.stride_tricks import sliding_window_view

def normalise_signal(signal):
  "Normalise signal by subtracting the mean and dividing by standard deviation"
  signal = signal - statistics.mean(signal)
  signal = signal / np.std(signal)
  return signal


def shift_signal(signal, shift):
  signal = np.append(signal, np.zeros(2 * shift))
  return np.roll(signal, shift)

arr = np.array([1,2,3,4,5])
arr2 = np.array([6,7,8,9,0])
print(sliding_window_view(arr, 2))
print(sliding_window_view(arr2, 2))
print(np.concatenate((sliding_window_view(arr2, 2), sliding_window_view(arr, 2)), axis=1))

# Retrieving the signal, consisting of 20 000 sample points measured at 1
window_size = 2001
hz = 1000
seconds = 10
samples = seconds * hz

# todo: merge thorax1 and thorax2 (see third example report)
######### REMEMBER TO GET RID OF THE [:samples]
t = np.linspace(0, seconds, samples, False)
t = range(samples)

thorax = np.loadtxt('data/thorax2.txt')[:samples]
abdomen = np.loadtxt('data/abdomen3.txt')[:samples]

thorax = normalise_signal(thorax)
abdomen_1 = normalise_signal(abdomen)

# Apply the highpass filter
sos = signal.butter(6, 20, 'highpass', fs=samples, output='sos')
thorax = signal.sosfilt(sos, thorax)
abdomen = signal.sosfilt(sos, abdomen_1)

thorax_windows = sliding_window_view(thorax, window_size)
print(thorax_windows.shape)
abdomen_targets = abdomen[1000:9000]

print(abdomen_targets.shape)


model = LinearRegression()

model.fit(thorax_windows, abdomen_targets)
prediction = model.predict(thorax_windows)
prediction = shift_signal(prediction, 1000)

# Plotting the original thorax1 signal
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(15, 5))

# Plot the original signal
ax1.plot(t, thorax)
ax1.set_title('Thorax')

ax2.plot(t, abdomen_1)
ax2.set_title('Abdomen')

ax3.plot(t, prediction)
ax3.set_title('prediction')

ax4.plot(t, abdomen - prediction)
ax4.set_title('Abdomen - prediction')

ax5.plot(t, abdomen)
ax5.set_title('Abdomen + filter ')

plt.tight_layout()
plt.show()
