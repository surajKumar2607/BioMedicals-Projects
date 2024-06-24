import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.fft as spf

eeg1 = np.loadtxt('C:\\Users\91801\\Documents\\Personal Docs\\MTECH\\IIT Kgp Course Sem 2\\4. Biomedicals\\Assignments\Assignment4\\eeg1.dat')
eeg2 = np.loadtxt('C:\\Users\91801\\Documents\\Personal Docs\\MTECH\\IIT Kgp Course Sem 2\\4. Biomedicals\\Assignments\Assignment4\\eeg2.dat')

plt.subplot(2,1,1)
plt.plot(eeg1)
plt.grid('True')

plt.subplot(2,1,2)
plt.plot(eeg2)
plt.grid('True')
plt.show()

def Welch_Method(x,M,S,w):
  if (w=='r'):
      w=   rect_window = np.ones(M)
  elif(w== 'h'):
      w= hanning_window = np.array([(0.54-0.46*np.cos(2*3.14*i/(M-1))) for i in range(M)])
  elif (w=='b'):
      w= blackman_window = np.array([0.42-0.5*np.cos(2*3.14*i/(M-1)) + 0.08*np.cos(4*3.14*i/(M-1)) for i in range(M)])

  N = len(x)
  K = N//S #Dividing total size by segments
  periodogram = []
  for i in range(K):
    xk = w * x[i:i+M]
    xk_fft =(abs(spf.fftshift(spf.fft(xk))) ** 2) / M
    periodogram.append(xk_fft)

  welch_psd = np.sum(periodogram[0:S],0)
  welch_psd = 10*np.log10(welch_psd)
    
  return welch_psd
 

def plot_fun(eeg1, k): 
    w1= Welch_Method(eeg1, 50, 3, k)
    w2= Welch_Method(eeg1, 150, 3, k)
    w3= Welch_Method(eeg1, 250, 3, k)
    
    plt.subplot(3,3,1)
    plt.grid('True')
    plt.plot(w1)
    plt.subplot(3,3,2)
    plt.grid('True')
    plt.plot(w2)
    plt.subplot(3,3,3)
    plt.grid('True')
    plt.plot(w3)
    
    w4= Welch_Method(eeg1, 50, 5, k)
    w5= Welch_Method(eeg1, 150, 5, k)
    w6= Welch_Method(eeg1, 250, 5, k)
    
    plt.subplot(3,3,4)
    plt.grid('True')
    plt.plot(w4)
    plt.subplot(3,3,5)
    plt.grid('True')
    plt.plot(w5)
    plt.subplot(3,3,6)
    plt.grid('True')
    plt.plot(w6)
    
    w7=Welch_Method(eeg1, 50, 10, k)
    w8= Welch_Method(eeg1, 150, 10, k)
    w9= Welch_Method(eeg1, 250, 10, k)
    
    plt.subplot(3,3,7)
    plt.grid('True')
    plt.plot(w7)
    plt.subplot(3,3,8)
    plt.grid('True')
    plt.plot(w8)
    plt.subplot(3,3,9)
    plt.grid('True')
    plt.plot(w9)
    plt.show()

plot_fun(eeg1, 'r')
plot_fun(eeg1, 'h')
plot_fun(eeg1, 'b')

plot_fun(eeg2, 'r')
plot_fun(eeg2, 'h')
plot_fun(eeg2, 'b')

