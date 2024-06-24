import numpy as np
import matplotlib.pyplot as plt
import scipy.io as si
import scipy.fft as sfft

# Load EMG signal from file
mat = si.loadmat('C:\\Users\\91801\\Documents\\Personal Docs\\MTECH\\IIT Kgp Course Sem 2\\4. Biomedicals\\Assignments\\Assignment3\\files\\pec1.mat',squeeze_me=True)
data1 = np.array(mat['car'])


mat = si.loadmat('C:\\Users\\91801\\Documents\\Personal Docs\\MTECH\\IIT Kgp Course Sem 2\\4. Biomedicals\\Assignments\\Assignment3\\files\\pec33.mat',squeeze_me=True)
data2 = np.array(mat['car'])


mat = si.loadmat('C:\\Users\\91801\\Documents\\Personal Docs\\MTECH\\IIT Kgp Course Sem 2\\4. Biomedicals\\Assignments\\Assignment3\\files\\pec52.mat',squeeze_me=True)
data3 = np.array(mat['car'])


plt.plot(data1[1000:3000])
plt.title("Original PEC1 Signal")
plt.grid('True')
plt.show()


plt.plot(data2[1000:3000])
plt.title("Original PEC33 Signal")
plt.grid('True')
plt.show()

plt.plot(data3[1000:3000])
plt.title("Original PEC52 Signal")
plt.grid('True')
plt.show()

#removing mean from each signals
mean1 = sum(data1)/len(data1)
mean2 = sum(data2)/len(data2)
mean3 = sum(data3)/len(data3)


mean_list1 = [mean1 for i in range(len(data1))]
zero_meean_sig1 = data1 - mean_list1


mean_list2 = [mean2 for i in range(len(data2))]
zero_meean_sig2 = data2 - mean_list2


mean_list3 = [mean3 for i in range(len(data3))]
zero_meean_sig3 = data3 - mean_list3


# calculating the FFT
sig_fft = sfft.fft(zero_meean_sig1)
N = len(sig_fft)
sig_fft[:((N//2) + 1)] = 0
freq_amp = 2*sig_fft
filtered_signal = sfft.ifft(freq_amp)
output_signal = abs(filtered_signal)
plt.plot(output_signal[1000:3000])
plt.title("Eneveogram of pec1")
plt.grid('True')
plt.show()

# calculating the FFT
sig_fft = sfft.fft(zero_meean_sig2)
N = len(sig_fft)
sig_fft[:((N//2) + 1)] = 0
freq_amp = 2*sig_fft
filtered_signal = sfft.ifft(freq_amp)
output_signal = abs(filtered_signal)
plt.plot(output_signal[1000:3000])
plt.title("Eneveogram of pec33")
plt.grid('True')
plt.show()

# calculating the FFT
sig_fft = sfft.fft(zero_meean_sig3)
N = len(sig_fft)
sig_fft[:((N//2) + 1)] = 0
filtered_signal = sfft.ifft(freq_amp)
output_signal = abs(filtered_signal)
plt.plot(output_signal[1000:3000])
plt.title("Eneveogram of pec52")
plt.grid('True')
plt.show()