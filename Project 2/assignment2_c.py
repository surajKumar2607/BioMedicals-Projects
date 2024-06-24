import math
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy
from scipy.signal import butter,filtfilt


########################## All the Functions used #################################

#Butterworth Low pass filter
#passing the signal to butterworth low pass filter
def butter_lowpass_filter(cutoff_freq, sampling_rate, order =8):
    nyqs = 0.5* sampling_rate
    normal_cutoff_freq  = cutoff_freq/nyqs
    b,a = butter(order, normal_cutoff_freq, btype= 'low', analog= False)
    return b,a

def butter_lowpass_after(cf, od, n_s_2):
    b1, a1 = butter_lowpass_filter(cf,200,od)
    filtered_data1 = filtfilt(b1,a1,n_s_2)
    filtered_data1 = np.array(filtered_data1)
    o_p = []
    i=0
    for i in range(filtered_data1.shape[1]):
        o_p.append(filtered_data1[0][i])
        i+=1
    plt.plot(o_p)
    plt.title("Response (Butterworth LPF)")
    plt.ylabel('Amplitude', color='b')
    plt.grid(True)
    plt.xlabel('Time', color='b')
    plt.show()
    return filtered_data1


#Butterworth High pass filter
#passing the signal to butterworth low pass filter
def butter_highpass_filter(cutoff_freq, sampling_rate, order =8):
    nyqs = 0.5* sampling_rate
    normal_cutoff_freq  = cutoff_freq/nyqs
    b,a = butter(order, normal_cutoff_freq, btype= 'high', analog= False)
    return b,a

def butter_highpass_after(cf, od, n_s_2):
    b1, a1 = butter_highpass_filter(cf,200,od)
    filtered_data1 = filtfilt(b1,a1,n_s_2)
    filtered_data1 = np.array(filtered_data1)
    o_p = []
    i=0
    for i in range(filtered_data1.shape[1]):
        o_p.append(filtered_data1[0][i])
        i+=1
    plt.plot(o_p)
    plt.title("Response (Butterworth HPF)")
    plt.ylabel('Amplitude', color='b')
    plt.grid(True)
    plt.xlabel('Time', color='b')
    plt.show()
    return filtered_data1

def diff_operator(x):
    y = []
    #y[n]= (1/8 )*(2*x[n]+ x[n-1]- x[n-3] -2x[n-4])
    #y.append((1/8 )*(2*x[0]))
    #y.append((1 / 8) * (2*x[1]+ x[0]))

    y.append(0)
    y.append(0)

    for n in range(x.size):
        if (n < 3):
            k = (1/8) *(2*x[n]+ x[n-1])
            #y.append(k)
            y.append(0)
        elif (n < 4):
            k = (1/8)* (2*x[n]+ x[n-1]- x[n-3])
            #y.append(k)
            y.append(0)
        else:
            k = (1/8 )*(2*x[n]+ x[n-1]- x[n-3] -2*x[n-4])
            y.append(k)
    y1 = np.array(y)
    return y1


def do_square(x):
    y =[]
    for n in range(x.shape[0]):
        k= x[n]*x[n]
        y.append(k)
    y1= np.array(y)
    return y1

def do_integration(x):
    y =[]
    for n in range(x.shape[0]):
        if (n<29):
            y.append(0)
        else:
            sum =0
            for i in range(n-29, n+1,1):
                sum += x[i]
            sum = sum/30
            y.append(sum)
    y1 = np.array(y)
    return y1



def detect_qrs(x,k):
    max= 0
    y =[]
    for i in range(x.shape[0]):
        if (x[i] > max):
            max = x[i]
    th = k*max
    for i in range(x.shape[0]):
        if (x[i] > th):
            y.append(x[i])
        else:
            y.append(th)
    y= np.array(y)
    return y


#calculating the average heart rate:
#To get the peak value we have to divide the region into subregion based on 50% threshold
def find_max(n1,n2,x):
    max =0
    for n in range(n1,n2+1,1):
        if (x[n]>max):
            max = x[n]
            nk = n
    return nk


def avg_heartbeat(x):
    max = 0
    y = []
    for i in range(x.shape[0]):
        if (x[i] > max):
            max = x[i]
    th = 0.5 * max #50%

    up_of_th = 0
    down_of_th = 1
    peaks = []
    max = 0
    flag = 0

    for n in range(x.shape[0]):
        if ((down_of_th==1)& (x[n]>th) & (up_of_th == 0 )):
            up_of_th = 1
            down_of_th =0
            n_start = n
            flag =0
        elif ((up_of_th==1)& (x[n]<th) &(down_of_th== 0)):
            down_of_th=1
            up_of_th =0
            n_stop = n
            flag = 1
        if (flag == 1):
            max = find_max(n_start,n_stop,x)
            flag =0
            peaks.append(max)
    peaks = np.array(peaks)
    sum =0
    for i in range(peaks.size-1) :
        sum += peaks[i+1]-peaks[i]
    avg = sum/(peaks.size-1)
    return avg


def do_all_the_process(data12, data11):
    plt.plot(data12)
    plt.title("Original signal")
    plt.grid(True)
    plt.show()
    data11l = butter_lowpass_after(11, 8, data11)
    data11h = butter_highpass_after(5, 8, data11l)
    #reshape data11h
    data11r = np.reshape(data11h, ( data11h.shape[1],1))
    data13 = diff_operator(data11r)
    plt.title("After diff")
    plt.plot(data13)
    plt.grid(True)
    plt.show()

    data14 = do_square(data13)
    plt.title("Do Square")
    plt.plot(data14)
    plt.grid(True)
    plt.show()

    data15 = do_integration(data14)

    plt.title("Do Integration")
    plt.plot(data15)
    plt.grid(True)
    plt.show()
    data16 = detect_qrs(data15, 0.95)
    plt.plot(data15)
    plt.plot(data16)
    plt.show()

    data16 = detect_qrs(data15, 0.8)
    plt.plot(data15)
    plt.plot(data16)
    plt.show()

    avg = avg_heartbeat(data15)
    avg_in_second = avg / 200
    ppm = 2*60 / avg_in_second
    print('Heart rate: ', ppm, 'ppm')




#Loading the data from dat file
data1 = loadmat(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\m_31.mat")
data2 = loadmat(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\m_32.mat")
data11 = data1["val"]
data21 = data2["val"]

#Reshaping to column vector
data12 = np.reshape(data11, (data11.shape[1],1))
data22 = np.reshape(data21, (data21.shape[1],1))


do_all_the_process(data12, data11)
do_all_the_process(data22, data21)