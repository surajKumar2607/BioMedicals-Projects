from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math as m


#Loading the data from dat file
with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-c3.dat", 'r') as f:
    data1 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-c4.dat", 'r') as f:
    data2 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-f3.dat", 'r') as f:
    data3 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-f4.dat", 'r') as f:
    data4 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-o1.dat", 'r') as f:
    data5 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-o2.dat", 'r') as f:
    data6 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-p3.dat", 'r') as f:
    data7 = f.read()

with open(r"C:\Users\91801\Documents\Personal Docs\MTECH\IIT Kgp Course Sem 2\4. Biomedicals\Assignments\Assignment2\Assignment 2_BMSEA\1-p4.dat", 'r') as f:
    data8 = f.read()

def convert_to_np(data):
    data1 =[]
    element = ''
    for i in range(len(data)):
        if (data[i]=='\n'):
            data1.append(float(element))
            element = ''
        else :
            element = element + data[i]

    data1 = np.array(data1)
    return data1


cdata1 = convert_to_np(data1)
cdata2 = convert_to_np(data2)
cdata3 = convert_to_np(data3)
cdata4 = convert_to_np(data4)
cdata5 = convert_to_np(data5)
cdata6 = convert_to_np(data6)
cdata7 = convert_to_np(data7)
cdata8 = convert_to_np(data8)

'''
print(cdata1.size)
print(cdata2.size)
print(cdata3.size)
print(cdata4.size)
print(cdata5.size)
print(cdata6.size)
print(cdata7.size)
print(cdata8.size)
'''

def print_fun (cdata1,cdata2,cdata3,cdata4,cdata5,cdata6,cdata7,cdata8):
    x = np.arange(0, 750, 1)
    fig, (a,b,c,d,e,f,g,h) = plt.subplots(8,1)

    a.plot(x,cdata1)
    a.grid(True)
    a.set_title('1')

    b.plot(x,cdata2)
    b.grid(True)
    b.set_title('2')

    c.plot(x,cdata3)
    c.grid(True)
    c.set_title('3')

    d.plot(x,cdata4)
    d.grid(True)
    d.set_title('4')

    e.plot(x,cdata5)
    e.grid(True)
    e.set_title('5')

    f.plot(x,cdata6)
    f.grid(True)
    f.set_title('6')

    g.plot(x,cdata7)
    g.grid(True)
    g.set_title('7')

    h.plot(x,cdata8)
    h.grid(True)
    h.set_title('8')
    plt.show()

#since the sampling rate is 102Hz so in 1 second there  are 102 data, so since the alpha rythem from 3 to 4 second
#Therefore the data is from 306 to 408 which is clearly visible in 2nd and 7th waveform
#let us choose the channel 2 as the alpha wave
print_fun (cdata1,cdata2,cdata3,cdata4,cdata5,cdata6,cdata7,cdata8)

alpha = []

for i in range(306, 409,1):
    alpha.append(cdata2[i]) #putting the channel name

ref_signal = alpha

def append_zero(odata, k):
    odata1 = []
    for j in range(odata.size):
        odata1.append(odata[j])
    for i in range(k):
        odata1.append(0.0)
    return odata1



def do_crosscorr(odata, rdata):
    odata1 = append_zero(odata, 102)
    cross = []
    for i in range(750):
        sum = 0
        for j in range(i,102+i,1):
            a = odata1[j]*rdata[j-i]
            sum += a
        cross.append(sum)
    cross = np.array(cross)
    #print(cross.size)
    return cross

cd1 = do_crosscorr(cdata1, ref_signal)
cd2 = do_crosscorr(cdata2, ref_signal)
cd3 = do_crosscorr(cdata3, ref_signal)
cd4 = do_crosscorr(cdata4, ref_signal)
cd5 = do_crosscorr(cdata5, ref_signal)
cd6 = do_crosscorr(cdata6, ref_signal)
cd7 = do_crosscorr(cdata7, ref_signal)
cd8 = do_crosscorr(cdata8, ref_signal)

print_fun (cd1,cd2,cd3,cd4,cd5,cd6,cd7,cd8)



