import numpy as np
import matplotlib.pyplot as plt

# Load EMG signal from file
emg = open('C:\\Users\\91801\\Documents\\Personal Docs\\MTECH\\IIT Kgp Course Sem 2\\4. Biomedicals\\Assignments\\Assignment3\\files\\emg.txt')



data = emg.read()

data1 =[]
num = ''

for i in data:
    if (i == ' ')or (i == '\n') :
        data1.append(float(num))
        num = ''
    else:
        num += i

time = []
value = []

for i in range(len(data1)):
    if (i % 2 == 0) :
        time.append(data1[i])
    else:
        value.append(data1[i])


time = np.array(time)
value = np.array(value)

## Plotting of the signal
plt.plot(value[:1000])
plt.title("Original EMG signal")
plt.grid('True')
plt.show()

#Since time gap is around 0.00025 sec so sampling freq = 1/0.00025
Sam_freq = 1/0.00025

def padBeforeSignal(sig,size):
  temp = [0 for i in range(size)]
  new_sig = temp + list(sig)
  return np.array(new_sig)


def RMS_rect(sig, window_size):
  window = np.array([1 for i in range(window_size)])
  padded_sig = padBeforeSignal(sig,window_size)
  filtered_sig = []
  for i in range(window_size,len(padded_sig)):
    filtered_sig.append(np.sqrt(sum((window*padded_sig[i:i-window_size:-1])**2)/window_size))
  return filtered_sig

def RMS_Hanning(y, Window_size):
  window = np.array([(0.5-0.5*np.cos(2*3.14159*i/(Window_size-1))) for i in range(Window_size)])
  padded_sig = padBeforeSignal(y,Window_size)
  sq_sig = padded_sig ** 2
  filtered_sig = []
  for i in range(Window_size,len(sq_sig)):
    filtered_sig.append(np.sqrt(sum((window*padded_sig[i:i-Window_size:-1])**2)/Window_size))
  return filtered_sig


temp = RMS_rect(value, 100)
plt.plot(temp[:1000])
plt.title("RMS value EMG signal with Rect window of size 100")
plt.grid('True')
plt.show()

temp = RMS_rect(value, 150)
plt.plot(temp[:1000])
plt.title("RMS value EMG signal with Rect window of size 150")
plt.grid('True')
plt.show()

temp = RMS_Hanning(value, 100)
plt.plot(temp[:1000])
plt.title("RMS value EMG signal with Hanning window of size 100")
plt.grid('True')
plt.show()

temp = RMS_Hanning(value, 150)
plt.plot(temp[:1000])
plt.title("RMS value EMG signal with Hanning window of size 150")
plt.grid('True')
plt.show()