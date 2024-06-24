import numpy as np
import matplotlib.pyplot as plt
import scipy.io as si


# Load EMG signal from file
mat = si.loadmat('C:\\Users\\91801\\Documents\\Personal Docs\\MTECH\\IIT Kgp Course Sem 2\\4. Biomedicals\\Assignments\\Assignment3\\files\\emg_dog2.mat',squeeze_me=True)
data = np.array(mat['emg'])


plt.plot(data)
plt.title("Original Signal")
plt.grid('True')
plt.show()

mean = (sum(data)/len(data))
mean_list = [mean for i in range(len(data))]
zero_meean_sig = data - mean_list

plt.plot(zero_meean_sig)
plt.title("Zero Mean Signal")
plt.grid('True')
plt.show()

def FWR(sig):
    for i in range(len(sig)):
        if (sig[i]<0):
            sig[i]= -1*sig[i]
    return sig

def HWR(sig):
    for i in range(len(sig)):
        if (sig[i]<0):
            sig[i]= 0
    return sig

fwr_sig = FWR(zero_meean_sig)
plt.plot(fwr_sig)
plt.title("Zero Mean Signal")
plt.grid('True')
plt.show()

from scipy import signal

# Define the cutoff frequency and sampling frequency
cutoff_freq = 15 # Hz
sampling_freq = 10000 # Hz

# Define the filter order
filter_order = 8

# Calculate the Nyquist frequency
nyquist_freq = 0.5 * sampling_freq

# Calculate the normalized cutoff frequency
normalized_cutoff_freq = cutoff_freq / nyquist_freq

# Create the Butterworth filter
a = signal.butter(filter_order, normalized_cutoff_freq, btype='lowpass', output = 'sos')
filtered_sig = signal.sosfiltfilt(a,zero_meean_sig)
plt.plot(filtered_sig)
plt.title("Full Rectified Signal Wave Rectified Signal")
plt.show()

fwr_sig = HWR(zero_meean_sig)
plt.plot(fwr_sig)
plt.title("Zero Mean Signal")
plt.grid('True')
plt.show()


# Define the cutoff frequency and sampling frequency
cutoff_freq = 15 # Hz
sampling_freq = 10000 # Hz

# Define the filter order
filter_order = 8

# Calculate the Nyquist frequency
nyquist_freq = 0.5 * sampling_freq

# Calculate the normalized cutoff frequency
normalized_cutoff_freq = cutoff_freq / nyquist_freq

# Create the Butterworth filter
a = signal.butter(filter_order, normalized_cutoff_freq, btype='lowpass', output = 'sos')
filtered_sig = signal.sosfiltfilt(a,zero_meean_sig)
plt.plot(filtered_sig)
plt.title("Half Rectified Signal Wave Rectified Signal")
plt.show()