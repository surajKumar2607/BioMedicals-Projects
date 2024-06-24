from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math as m


#Loading the data from matlab file
data = loadmat(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment1\data_ecg_original")
x1 = data["ecg_original"]
x2 = np.reshape(x1, (x1.shape[1],1))
# from row vector to column vector i.e. (1, 650000) to (650000,1)


#Creating the power line interference
Amp = 0.15
freq = 61
time = np.arange(650000) #same sample number as x2
noise = []
for n in time:
    sin_noise = Amp * m.sin(2*m.pi*freq*n/300) #fs = 300 hz therefore in 1 sec 300 sample and 61 peack per sec
    noise.append(sin_noise)


#Adding noise to signal
n_s = []
for n in time:
    n_s.append(noise[n] + x2[n])

# for Displaying the sliced signal + noise, uncomment the below code
#n_s = n_s[0:500]
#plt.plot(n_s)
#plt.show()


#Designing Notch filter

f0= 61
fs=  300
w0 = 2*m.pi* f0 / fs
z1 = complex(m.cos(w0) , + m.sin(w0))
z2 = complex (m.cos(w0), - m.sin(w0))

#filtered signal:
y= []

for n in time:
    if (n==0):
        y.append(n_s[n])
    elif(n==1):
        y.append( n_s[n]- (z1+z2)*n_s[n-1])
    else:
        y.append( n_s[n]- (z1+z2)*n_s[n-1] + (z1*z2)*n_s[n-2])

#taking the real value from the filtered o/p
y_r = []
for n in time:
    y_r.append(y[n].real)


#for poles
rad = input(" Enter the pole radius (0.8 - 0.95): ")
rad = float(rad)
z1p = rad* z1
z2p = rad * z2

#filtered signal with poles:
y1= []

for n in time:
    if (n==0):
        y1.append(n_s[n])
    elif(n==1):
        y1.append( n_s[n] + (z1p+z2p)*y1[n-1]- (z1+z2)*n_s[n-1])
    else:
        y1.append( n_s[n] - (z1+z2)*n_s[n-1] + (z1*z2)*n_s[n-2] + (z1p+z2p)*y1[n-1] - (z1p*z2p)*y1[n-2])

y_rp = []
for n in time:
    y_rp.append(y1[n].real)

y_r = y_r[0:1000]
x2 = x2[0:1000]
n_s = n_s[0:1000]
y_rp = y_rp[0:1000]


plt.plot(x2)
plt.title('pure ecg signal')
plt.show()

plt.plot(n_s, 'b')
plt.title('noise with ecg signal')
plt.show()

plt.plot(y_r, 'r')
plt.title('filtered output only with notch filter')
plt.show()

plt.plot(n_s, 'b')
plt.title('comparison between noisy and filtered signal')
plt.plot(y_r, 'r')
plt.show()

plt.plot(y_rp, 'g')
plt.title('filtered output with zeroes and poles')
plt.show()

plt.plot(n_s, 'b')
plt.title('comparison 1')
plt.plot(y_rp, 'g')
plt.show()

plt.plot(y_r, 'r')
plt.title('comparison 2')
plt.plot(y_rp, 'g')
plt.show()




