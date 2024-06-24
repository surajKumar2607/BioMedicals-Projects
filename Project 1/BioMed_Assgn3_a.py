from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy
from scipy.signal import butter,filtfilt

#Loading the data from matlab file
data = loadmat(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment1\ecg_noisy_high")
n_s = data["ecg"]


#filtered signal with three point central difference operator:
y= []
time = np.arange(n_s.shape[0])

for n in time:
    if (n==0):
        y.append(n_s[n]/4)
    elif(n==1):
        y.append( n_s[n]/4)
    else:
        y.append(( n_s[n]- n_s[n-2])/4)

#printing the output from three point central difference operator
n_s_1 = n_s[0:5000]
y1 = y[0:5000]
plt.plot(n_s_1, 'b')
plt.show()

plt.plot(y1, 'g')
plt.title("Response green: filtered, blue: original")
plt.ylabel('Amplitude', color='b')
plt.xlabel('Time', color='b')
plt.show()

#passing the noisy signal from butterworth high pass filter
def butter_highpass_filter(cutoff_freq, sampling_rate, order =2):
    nyqs = 0.5* sampling_rate
    normal_cutoff_freq  = cutoff_freq/nyqs
    b,a = butter(order, normal_cutoff_freq, btype= 'high', analog= False)
    return b,a

def butter_highpass_after(cf, od):
    n_s_2 = np.reshape(n_s, (1, n_s.shape[0]))
    b1, a1 = butter_highpass_filter(cf,300,od)
    filtered_data1 = filtfilt(b1,a1,n_s_2)
    filtered_data1 = np.array(filtered_data1)
    o_p = []
    i=0
    for i in range(filtered_data1.shape[1]):
        o_p.append(filtered_data1[0][i])
        i+=1
    plt.plot(o_p[0:5000])
    plt.title("Response (Butterworth HPF)")
    plt.ylabel('Amplitude', color='b')
    plt.xlabel('Time', color='b')
    plt.show()


butter_highpass_after(3,3)
butter_highpass_after(5,5)
butter_highpass_after(8,8)

