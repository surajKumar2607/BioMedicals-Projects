from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft
import scipy.signal as signal
import statistics

#Loading the data from matlab file
data = loadmat(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment1\data_ecg_original")
x1 = data["ecg_original"]
x2 = np.reshape(x1, (x1.shape[1],1))


#Creating noise
p_sig = (np.linalg.norm(x2)/math.sqrt(len(x2))) * (np.linalg.norm(x2)/math.sqrt(len(x2))) #power of signal
p_noise = p_sig * pow(10, (-0.5/10)) #power of noise
amp_noise = math.sqrt(p_noise) #amplitude of noise
noise = amp_noise * np.random.normal(0,1,len(x2)) #noise


orig_ecg = x2
rnd_noise = noise

#Adding signal with noise
ecg_noisy_wgn = []
for i in range (len(x2)):
    ecg_noisy_wgn.append(x2[i]+noise[i])

#calculating the SNR of the input signal
snr = 20*np.log10(np.linalg.norm(orig_ecg)/np.linalg.norm(rnd_noise))
print("SNR of input signal (before filtering) :", snr)

ecg_noisy_wgn = np.array(ecg_noisy_wgn)
ecg_noisy_wgn = ecg_noisy_wgn[0:(650000-1)]
ecg_noisy_wgn = np.reshape(ecg_noisy_wgn, (1, (650000-1)))

def fltr_func(fc, n):
    fsample = 360
    # Filter coefficient for the Butterworth filter
    b, a = signal.butter(n, float(fc) / (float(fsample) / 2), 'low', analog= False) #2nd parameter is normalized frequency
    '''
    # frequency spectrum of the filter, to display it remove the comment
    w, h = signal.freqz(b, a)
    # Magnitude Spectrum
    plt.plot(w * (180 / np.pi), 20 * np.log10(abs(h)), 'b')
    plt.title('Frequency Response (Butterworth Cutoff freq =%i)' % fc)
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hertz]')
    plt.show()
    # Phase Spectrum
    angles = np.unwrap(np.angle(h))
    plt.plot(w * (180 / np.pi), angles, 'g')
    plt.title('Phase Response (Butterworth Cutoff freq =%i)' % fc)
    plt.ylabel('Angle (radians)', color='g')
    plt.xlabel('Frequency [Hertz]')
    # plt.savefig('Phase Response.jpeg')#to save in local
    plt.show()
    '''

    # filter output
    out = signal.filtfilt(b, a, ecg_noisy_wgn)
    out = np.array(out)
    out = np.reshape(out, (650000-1,1))
    '''
    #to display the filtered output uncomment
    plt.plot(out)
    plt.show()
    '''

    out1 = []
    sig = []
    out2 = []
    count = 0
    for i in out:
        out1.append(x2[count][0] - out[count][0])
        sig.append(x2[count][0])
        out2.append(out[count][0]- x2[count][0])
        count+=1

    #calculating the SNR and SDR
    res1 = statistics.variance(out2)
    res2 = statistics.variance(sig)
    res3 = statistics.variance(out1)
    print ("Improvemnet with fc = ",fc, " and n= ", n, ": ", (10 * res2/res1)-snr  )
    print("SDR: =", 10*res2/res3)


fltr_func(30, 6)
fltr_func(40, 6)
fltr_func(50, 8)
fltr_func(70, 8)