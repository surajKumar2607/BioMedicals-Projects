import math
import numpy as np
import matplotlib.pyplot as plt

#Loading the data from dat file
with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\21.dat", 'r') as f:
    data1 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\22.dat", 'r') as f:
    data2 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\23.dat", 'r') as f:
    data3 = f.read()

def convert_to_np(data):
    data1 =[]
    ch1 =[]
    ch2 =[]
    ch3=[]
    element = ''
    for i in range(len(data)-1):

        if ((data[i]=='\n')or (data[i]=='\t') ):
            data1.append(float(element))
            element = ''

        else :
            element = element + data[i]
    data1 = np.array(data1)

    for j in range(data1.size):
        a= data1[j]
        b = (a)
        if (j%3 == 0):
            ch1.append(b)
        elif (j % 3 == 1):
            ch2.append(b)
        else :
            ch3.append(b)
    ch1 = np.array(ch1)
    ch2 = np.array(ch2)
    ch3 = np.array(ch3)
    return ch1,ch2,ch3


ch11,ch12,ch13 = convert_to_np(data1)
ch21,ch22,ch23 = convert_to_np(data2)
ch31,ch32,ch33 = convert_to_np(data3)


s = 5000
s0 = 1000
ch11 = ch11[s0:s]
ch12 = ch12[s0:s]
ch13 = ch13[s0:s]

ch21 = ch21[s0:s]
ch22 = ch22[s0:s]
ch23 = ch23[s0:s]

ch31 = ch31[s0:s]
ch32 = ch32[s0:s]
ch33 = ch33[s0:s]

def print_fun (cdata1,cdata2,cdata3):
    x = np.arange(0, cdata1.size/900, 1/900)
    x1 = np.arange(0, cdata3.size / 900, 1 / 900)
    fig, (a,b,c) = plt.subplots(3,1)

    a.plot(x, cdata1)
    a.grid(True)
    a.set_title('1')

    b.plot(x, cdata2)
    b.grid(True)
    b.set_title('2')

    c.plot(x1, cdata3)
    c.grid(True)
    c.set_title('3')
    plt.show()

'''
print_fun(ch11,ch12,ch13)
print_fun(ch21,ch22,ch23)
print_fun(ch31,ch32,ch33)
'''

#formula y(n) = x(n+1) - 2*x(n) + x(n-1)
def do_second_der(x):
    y=[]
    y.append(0)
    for n in range(1,x.size-1):
        y_n = x[n+1]- 2*x[n] + x[n-1]
        y.append(y_n)
    y = np.array(y)
    return y

#formula y[n]= x[n] - x[n-1]
def do_second_der2(x):
    y=[]
    y.append(0)
    for n in range(1,x.size):
        y_n = x[n]- x[n-1]
        y.append(y_n)
    y = np.array(y)
    return y

def do_MA(x):
    y=[]
    for n in range(x.size):
        if((n-32)<0):
            sum= 0
            for i in range(n+1):
                sum += x[i]
            y_n =  sum/(n+1)
            y.append(0)
        else:
            sum = 0
            for i in range(n-32,n,1):
                sum += x[i]
            y_n = sum / 32
            y.append(y_n)
    y = np.array(y)
    return y

ch132 = do_MA(do_second_der2(do_MA(do_second_der2(ch13))))
ch232 = do_MA(do_second_der2(do_MA(do_second_der2(ch23))))
ch332 = do_MA(do_second_der2(do_MA(do_second_der2(ch33))))

'''
print_fun(ch11,ch13,ch132)
print_fun(ch21,ch23,ch232)
print_fun(ch31,ch33,ch332)

'''
#Since the sampling rate is 900 Hz therefore for +/- 20 ms, so we need to check +/- 18 samples around 2nd peak will give S2


#Pan-Tompkins method
def difference_equation(x):
    y = np.zeros(len(x)) # initialize output signal y with zeros
    y[0] = 0 # set y[0] to 0
    #y[1] = x[0] # set y[1] to x[0]
    for n in range(12, len(x)):
        y[n] = 2*y[n-1] - y[n-2] + (1/32)*(x[n] - 2*x[n-6] + x[n-12])
    return y


def pt_lowpass(x):
    y = []
    #y[n]= 2*y[n-1]- y[n-2]+ (1/32)[x[n]- 2*x[n-6] + x[n-12]
    #y.append(x[0]/32)
    y.append(0)
    #y.append(2*y[0] + (1/32)*(x[1]))
    y.append(0)
    for n in range(x.size):
        if (n<6):
            k = 2*y[n-1]- y[n-2]+ (1/32)*(x[n])
            y.append(0)
        elif (n<12):
            k = 2*y[n-1]- y[n-2]+ (1/32)*(x[n]- 2*x[n-6])
            y.append(0)
        else:
            k = 2*y[n-1]- y[n-2]+ (1/32)*(x[n]- 2*x[n-6] + x[n-12])
            y.append(k)
    y = np.array(y)
    return y

def pt_bandpass(x):
    y = []
    #y[n]= x[n-16] -(1/32)(y[n-1] + x[n] - x[n-32] )
    #y.append(-(1/32)*(x[0]))
    y.append(0)
    for n in range(x.size):
        if (n < 16):
            k = -(1/32)*(y[n-1] + x[n])
            y.append(0)
        elif (n < 32):
            k = x[n-16] -(1/32)*(y[n-1] + x[n])
            y.append(0)
        else:
            k = x[n-16] -(1/32)*(y[n-1] + x[n] - x[n-32] )
            y.append(k)
    y = np.array(y)
    return y


def pt_highpass(x):
    y = np.zeros(len(x))
    #y[n]= y[n-1] +x[n] - x[n-32]
    #y.append((x[0]))
    #y.append(0)
    for n in range(32,len(x)):
        if (n < 32):
            k = y[n-1] +x[n]
            y.append(0)
        else:
            y[n] = y[n-1] +x[n] - x[n-32]
            #y.append(k)
    #y = np.array(y)
    return y

def diff_operator2(x):
    y = []
    #y[n]= (1/8 )*(2*x[n]+ x[n-1]- x[n-3] -2x[n-4])
    #y.append((1/8 )*(2*x[0]))
    y.append(0)
    #y.append((1 / 8) * (2*x[1]+ x[0]))
    y.append(0)

    for n in range(x.size):
        if (n < 3):
            k = (1/8) *(2*x[n]+ x[n-1])
            y.append(0)
        elif (n < 4):
            k = (1/8)* (2*x[n]+ x[n-1]- x[n-3])
            y.append(0)
        else:
            k = (1/8 )*(2*x[n]+ x[n-1]- x[n-3] -2*x[n-4])
            y.append(k)
    y = np.array(y)
    return y

def do_MA2(x):
    y=[]
    for n in range(x.size):
        if((n-30)<0):
            sum= 0
            for i in range(n+1):
                sum += x[i]
            y_n =  sum/(n+1)
            y.append(0)
        else:
            sum = 0
            for i in range(n-30,n,1):
                sum += x[i]
            y_n = sum / 30
            y.append(y_n)
    y = np.array(y)
    return y

def f900_to_f200(x):
    m = np.arange(0, (math.ceil(x.shape[0]/4.5)+1)/0.5, 0.5)
    m[0]= x[0]
    for n in range(x.size):
        m[math.ceil(n/4.5)]= float(x[n])
    m = np.array(m)
    return m

def do_square2(x):
    y =[]
    for n in range(x.shape[0]):
        k= x[n]*x[n]
        y.append(k)
    y= np.array(y)
    return y

#################################################################

def diff_operator(x):
    y = np.zeros(len(x))
    #y[n]= (1/8 )*(2*x[n]+ x[n-1]- x[n-3] -2x[n-4])
    #y.append((1/8 )*(2*x[0]))
    #y.append((1 / 8) * (2*x[1]+ x[0]))


    for n in range(4, len(x)):
        if (n < 3):
            k = (1/8) *(2*x[n]+ x[n-1])
            #y.append(k)
            y.append(0)
        elif (n < 4):
            k = (1/8)* (2*x[n]+ x[n-1]- x[n-3])
            #y.append(k)
            y.append(0)
        else:
            y[n] = (1/8 )*(2*x[n]+ x[n-1]- x[n-3] -2*x[n-4])


    return y


def do_square(x):
    y =[]
    for n in range(x.shape[0]):
        k= x[n]*x[n]
        y.append(k)
    y1= np.array(y)
    return y1


def do_900_to_200(signal_900hz):
    from scipy.signal import resample
    import numpy as np

    resampled_signal = resample(signal_900hz, int(len(signal_900hz) * (200 / 900)))
    resampled_signal = np.array(resampled_signal)
    return resampled_signal


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


##############################################################

def Pan_Tompkin(x):
    m= x
    m = do_900_to_200(m)
    m= difference_equation(m)
    m = pt_highpass(m)
    m = diff_operator(m)
    m= do_square(m)
    m = do_integration(m)
    return m

#Adaptive thresholding


pt1 = Pan_Tompkin(ch22)
k = do_900_to_200(ch22)
plt.plot(k)
plt.plot(pt1)
plt.show()

#######################
ch2321 = do_900_to_200(ch232)

ch211 = do_900_to_200(ch21)
print_fun(ch211,ch2321,pt1)



#initial threshold value
th1 = 10
th2 = 0.1
npki = 0.5

def find_local_peaks(sequence):
    peaks = []
    for i in range(1, len(sequence)-1):
        if (sequence[i] > sequence[i-1] and sequence[i] > sequence[i+1]):
            peaks.append(sequence[i])
    return peaks

def find_max(x):
    max =0
    for n in range(len(x)):
        if x[n]>max :
            max = x[n]
    return max

peaks = find_local_peaks(pt1)
spki = find_max(peaks)

for k in range(len(peaks)):
    if(peaks[k] > th1):
        spki = 0.125*peaks[k] + 0.875*spki
    else:
        npki = 0.125*peaks[k] + 0.875*npki

    th1 = npki + 0.25*(spki - npki)
    th2 = 0.5* th1

print(th1, th2)






