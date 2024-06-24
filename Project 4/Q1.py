import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.io as spio

x = np.linspace(0,4.999,5000)
print (len(x))
y = np.sin(2*np.pi*70*x) + np.cos(2*np.pi*80*x)

plt.plot(y[0:500])
plt.grid("True")
plt.show()


fs = 1000
nyq_rate = 0.5*fs

#ns window size, window = window type
def do_periodogram(ns,window):
  x = np.zeros(ns) #ns number of zeroes [0(1 time), 0(2 time), ..., 0(ns times)]
  fz,Pzz = signal.periodogram(x,fs,window) #function calculates the power spectral density (PSD) of a signal using a periodogram.
      #fx: The frequencies at which the PSD is estimated. Pxx: The PSD of the input signal.
  Pyy = np.zeros(fz.size)
  #print(window.size)
  for j in np.arange(y.size-ns): #running for each from j=0 to j = y.size -ns, shifting the window
    f,Pyy_t = signal.periodogram(y[j:j+ns],fs,window)
    Pyy += Pyy_t #Adding each segment
  return f,Pyy

tm1 = 0.5
ns1 = int((tm1+1.0/fs)*fs)
tm2 = 1.5
ns2 = int((tm2+1.0/fs)*fs)

wdw1 = signal.get_window('boxcar',ns1) #to generate a boxcar window of length ns1 and assign it to the variable wdw1
f1,Pyy1 = do_periodogram(ns1,wdw1)
plt.figure
plt.semilogy(f1,Pyy1)
plt.ylim([1e-1, 1e4])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V^2/Hz]')
plt.grid(True)
plt.show()

wdw2 = signal.get_window('boxcar',ns2)
f2,Pyy2 = do_periodogram(ns2,wdw2)
plt.figure
plt.semilogy(f2,Pyy2)
plt.ylim([1e-3, 1e4])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V^2/Hz]')
plt.grid(True)
plt.show()

tm1 = 1.0
ns1 = int((tm1+1.0/fs)*fs)
tm2 = 2.0
ns2 = int((tm2+1.0/fs)*fs)

wdw3 = signal.get_window('hanning',ns1)
f3,Pyy3 = do_periodogram(ns1,wdw3)
plt.figure
plt.semilogy(f3,Pyy3)
plt.ylim([1e-7, 1e4])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V^2/Hz]')
plt.grid(True)
plt.show()

wdw4 = signal.get_window('hanning',ns2)
f4,Pyy4 = do_periodogram(ns2,wdw4)
plt.figure
plt.semilogy(f4,Pyy4)
plt.ylim([1e-11, 1e4])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V^2/Hz]')
plt.grid(True)
plt.show()


print('--------------------------------------------')
print('variance of first fft = ', np.std(Pyy1))
print('kurtosis of first fft = ', st.kurtosis(Pyy1))
print('skewness of first fft = ', st.skew(Pyy1))
print('--------------------------------------------')
print('variance of second fft = ', np.std(Pyy2))
print('kurtosis of second fft = ', st.kurtosis(Pyy2))
print('skewness of second fft = ', st.skew(Pyy2))
print('--------------------------------------------')
print('variance of Third fft = ', np.std(Pyy3))
print('kurtosis of Third fft = ', st.kurtosis(Pyy3))
print('skewness of Third fft = ', st.skew(Pyy3))
print('--------------------------------------------')
print('variance of Fourth fft = ', np.std(Pyy4))
print('kurtosis of Fourth fft = ', st.kurtosis(Pyy4))
print('skewness of Fourth fft = ', st.skew(Pyy4))
print('--------------------------------------------')

