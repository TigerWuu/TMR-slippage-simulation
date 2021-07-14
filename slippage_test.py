import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.animation as animation
import threading
import multiprocessing as mp
import time

import Trajectory as T

""" 
-------------------------
Author : Jing-Shiang Wuu
Date : 2021/6/25
Institution : National Taiwan University 
Department : Bio-Mechatronics Engineering
Status : Senior
-------------------------

Description:
    This is a simulation with TMR robot system. We make the robot track the trajectory we desired
    while the influence of the slippage will bring to a bad result. We want to compensate the loss of the wheel speed
    caused by the slippage and find out the solution by adpative slippage estimation and neural network slippage 
    prediction. This is a demostrate of the NN slippage prediction method.

Instruction:
    1. There are four trajectory simulation ,saparately:
        # reference trajectory
        # trajectory without slippage compensation
        # trajectory with slippage compensation offline
        # trajectory with slippage compensation online

    2. you have to run "slippage _train.py" first to get the NN model "slippage_predict"
    
    3. "ra" can be changed to modified the slippage when the time reach the time_step/3
        # ra = 1 ---> slippage remains
        # ra > 1 ---> slippage = slippage * ra

    4. "online" can switch the training status
        # online = True ---> online & offline
        # online = False ---> offline only

    5. Multithread.
        # multithread = True : about 43s
        # multithread = False : about 50s  
         
"""

### simulation step
time_step = 200

### car width
L = 20

### reference trajectory
trajectory = T.Trajectory("circle",time_step)
t,xr,yr,vxr,vyr,vr,thetar,omegar = trajectory.generator()

### real trajecotory
x = np.zeros(time_step)
y = np.zeros(time_step)
vx = np.zeros(time_step)
vy = np.zeros(time_step)
# v = np.zeros(time_step) 
theta = np.zeros(time_step)
omega = np.zeros(time_step)
wl = np.zeros(time_step)
wr = np.zeros(time_step)
s1 = np.zeros(time_step)
s2 = np.zeros(time_step)

### real trajectory after slippage compensation 
xp = np.zeros(time_step)
yp = np.zeros(time_step)
vxp = np.zeros(time_step)
vyp = np.zeros(time_step)

thetap = np.zeros(time_step)
omegap = np.zeros(time_step)
wlp = np.zeros(time_step)
wrp = np.zeros(time_step)
s1p = np.zeros(time_step)
s2p = np.zeros(time_step)

s1pt = np.zeros(time_step)
s2pt = np.zeros(time_step)

### real trajectory after slippage compensation online
xpo = np.zeros(time_step)
ypo = np.zeros(time_step)
vxpo = np.zeros(time_step)
vypo = np.zeros(time_step)

thetapo = np.zeros(time_step)
omegapo = np.zeros(time_step)
wlpo = np.zeros(time_step)
wrpo = np.zeros(time_step)
s1po = np.zeros(time_step)
s2po = np.zeros(time_step)

s1pot = np.zeros(time_step)
s2pot = np.zeros(time_step)

### controller output
vc = 0
vcp = 0
vcpo = 0
omegac = 0
omegacp = 0
omegacpo = 0

wrc = np.zeros(time_step)
wlc = np.zeros(time_step)
wrcp = np.zeros(time_step)
wlcp = np.zeros(time_step)
wrcpo = np.zeros(time_step)
wlcpo = np.zeros(time_step)

wrc_dot = np.zeros(time_step)
wlc_dot = np.zeros(time_step)
wrcp_dot = np.zeros(time_step)
wlcp_dot = np.zeros(time_step)
wrcpo_dot = np.zeros(time_step)
wlcpo_dot = np.zeros(time_step)

### error & error gain
e1 = np.zeros(time_step)
e2 = np.zeros(time_step)
e3 = np.zeros(time_step)
e1p = np.zeros(time_step)
e2p = np.zeros(time_step)
e3p = np.zeros(time_step)
e1po = np.zeros(time_step)
e2po = np.zeros(time_step)
e3po = np.zeros(time_step)

k1 = 0
k2 = 0
k3 = 0
k1p = 0
k2p = 0
k3p = 0
k1po = 0
k2po = 0
k3po = 0

### initial condition
x0 = 0.1
y0 = -0.1
theta0 = 1/4*np.pi

### slippage changes ratio
ra = 1.3

### online/offline
online = True

### multithread on/off
multithread = True


### load the Network
def load_network():
    slippage = keras.models.load_model("slippage_predict")
    return slippage

### slippage ratio empirical equation
def slippage_empirical(lv,rv,i):
    global time_step, ra
    alpha_b = -0.15
    beta_b = -0.63
    alpha_s = 0.07
    beta_s = -0.68
    if lv >= rv:
        R =  0.5*L*(lv+rv)/(lv-rv)
        alphal = alpha_b
        betal = beta_b
        alphar = alpha_s
        betar = beta_s
    else:
        R =  0.5*L*(lv+rv)/(rv-lv)
        alphal = alpha_s
        betal = beta_s
        alphar = alpha_b
        betar = beta_b
    if R>0:
        if i < time_step/3:
            i1 = alphal*np.exp(betal*R)
            i2 = alphar*np.exp(betar*R)
        else:
            i1 = ra*alphal*np.exp(betal*R)
            i2 = ra*alphar*np.exp(betar*R)
    else:
        i1 = 0
        i2 = 0
    return i1,i2
### without slippage compensation
def without_compensation():
    for i in range(time_step):
        if thetar[i] < 0:
            thetar[i] += 2*np.pi
        if i == 0:
            e1[i] = xr[i]
            e2[i] = yr[i]
            e3[i] = thetar[i]
        else:
            e1[i] = xr[i] - x[i-1]
            e2[i] = yr[i] - y[i-1]
            e3[i] = thetar[i] - theta[i-1]

        k1 = 2 * (omegar[i]**2 + 14*vr[i]**2)**0.5
        k2 = 14*np.abs(vr[i])
        k3 = k1

        ### state feedback & feedforward controller
        vc = vr[i]*np.cos(e3[i]) + k1*e1[i]
        omegac = omegar[i] + k2*np.sign(vr[i])*e2[i] + k3*e3[i]

        ###　inverse kinematic
        wlc[i] = vc*(1/(1-s1[i])) - omegac*L/(2*(1-s1[i]))
        wrc[i] = vc*(1/(1-s2[i])) + omegac*L/(2*(1-s2[i]))
        if i == 0:
            wlc_dot[i] = wlc[i]
            wrc_dot[i] = wrc[i]
        else:
            wlc_dot[i] = wlc[i] - wlc[i-1]
            wrc_dot[i] = wrc[i] - wrc[i-1]

        # slippage ratio empirical equation
        s1[i],s2[i] = slippage_empirical(wlc[i],wrc[i],i)

        ###　forward kinematic
        wl[i] = wlc[i]*(1-s1[i])
        wr[i] = wrc[i]*(1-s2[i])
        omega[i] = 1/L*wr[i] - 1/L*wl[i] #ccw > 0
        if  i == 0 :
            x[0] = x0
            y[0] = y0
            theta[0] =theta0
        else:
            x[i] += x[i-1] + vx[i-1]
            y[i] += y[i-1] + vy[i-1] 
            theta[i] = theta[i-1] + omega[i]
        vx[i] = 1/2*(np.cos(theta[i])*wl[i]+ np.cos(theta[i])*wr[i])
        vy[i] = 1/2*(np.sin(theta[i])*wl[i]+ np.sin(theta[i])*wr[i])
        # v[i] = (vx[i]**2 + vy[i]**2)**0.5

def with_compensation():
    global time_step, ra
    ### load the Network
    slippage = load_network()
   
    ### with slippage compensation
    for i in range(time_step):
        if thetar[i] < 0:
            thetar[i] += 2*np.pi
        if i == 0:
            s1p[i] = 0
            s2p[i] = 0
            e1p[i] = xr[i] - xp[i]
            e2p[i] = yr[i] - yp[i]
            e3p[i] = thetar[i] - thetap[i]
        else:
            x_online = [[wrcp[i-1],wlcp[i-1],wrcp_dot[i-1],wlcp_dot[i-1],wrp[i-1],wlp[i-1]]]
            sp = slippage.predict(x_online) 
            s1p[i] = sp[0][0]
            s2p[i] = sp[0][1]
            e1p[i] = xr[i] - xp[i-1]
            e2p[i] = yr[i] - yp[i-1]
            e3p[i] = thetar[i] - thetap[i-1]

        k1p = 2 * (omegar[i]**2 + 14*vr[i]**2)**0.5
        k2p = 14*np.abs(vr[i])
        k3p = k1p

        ### state feedback & feedforward controller
        vcp = vr[i]*np.cos(e3p[i]) + k1p*e1p[i]
        omegacp = omegar[i] + k2p*np.sign(vr[i])*e2p[i] + k3p*e3p[i]

        ###　inverse kinematic
        wlcp[i] = vcp*(1/(1-s1p[i])) - omegacp*L/(2*(1-s1p[i]))
        wrcp[i] = vcp*(1/(1-s2p[i])) + omegacp*L/(2*(1-s2p[i]))

        # slippage ratio empirical equation
        s1pt[i],s2pt[i] = slippage_empirical(wlcp[i],wrcp[i],i)

        ### forward kinematic
        wlp[i] = wlcp[i]*(1-s1pt[i])
        wrp[i] = wrcp[i]*(1-s2pt[i])
        omegap[i] = 1/L*wrp[i] - 1/L*wlp[i] 

        if i == 0:
            wrcp_dot[i] = wrcp[i]
            wlcp_dot[i] = wlcp[i]
            xp[0] = x0
            yp[0] = y0
            thetap[0] =theta0
        else:
            wrcp_dot[i] = wrcp[i] - wrcp[i-1]
            wlcp_dot[i] = wlcp[i] - wlcp[i-1]

            xp[i] += xp[i-1] + vxp[i-1]
            yp[i] += yp[i-1] + vyp[i-1] 
            thetap[i] = thetap[i-1] + omegap[i]

        vxp[i] = 1/2*(np.cos(thetap[i])*wlp[i]+ np.cos(thetap[i])*wrp[i])
        vyp[i] = 1/2*(np.sin(thetap[i])*wlp[i]+ np.sin(thetap[i])*wrp[i])

def with_compensation_online():
    global time_step, ra
    ### load the Network
    slippage = load_network()

    ### with slippage compensation online
    if online == True:
        for i in range(time_step):
            if thetar[i] < 0:
                thetar[i] += 2*np.pi
            if i == 0:
                s1po[i] = 0
                s2po[i] = 0
                e1po[i] = xr[i] - xpo[i]
                e2po[i] = yr[i] - ypo[i]
                e3po[i] = thetar[i] - thetapo[i]
            else:
                
                x_online = [[wrcpo[i-1],wlcpo[i-1],wrcpo_dot[i-1],wlcpo_dot[i-1],wrpo[i-1],wlpo[i-1]]]
                y_online = [[s1pot[i-1],s2pot[i-1]]]
                slippage.fit(x_online , y_online , epochs=3 ,batch_size = 1)
                spo = slippage.predict(x_online) 
                s1po[i] = spo[0][0]
                s2po[i] = spo[0][1]
                e1po[i] = xr[i] - xpo[i-1]
                e2po[i] = yr[i] - ypo[i-1]
                e3po[i] = thetar[i] - thetapo[i-1]

            k1po = 2 * (omegar[i]**2 + 14*vr[i]**2)**0.5
            k2po = 14*np.abs(vr[i])
            k3po = k1po

            ### state feedback & feedforward controller
            vcpo = vr[i]*np.cos(e3po[i]) + k1po*e1po[i]
            omegacpo = omegar[i] + k2po*np.sign(vr[i])*e2po[i] + k3po*e3po[i]

            ###　inverse kinematic
            wlcpo[i] = vcpo*(1/(1-s1po[i])) - omegacpo*L/(2*(1-s1po[i]))
            wrcpo[i] = vcpo*(1/(1-s2po[i])) + omegacpo*L/(2*(1-s2po[i]))

            # slippage ratio empirical equation
            s1pot[i],s2pot[i] = slippage_empirical(wlcpo[i],wrcpo[i],i)

            ### forward kinematic
            wlpo[i] = wlcpo[i]*(1-s1pot[i])
            wrpo[i] = wrcpo[i]*(1-s2pot[i])
            omegapo[i] = 1/L*wrpo[i] - 1/L*wlpo[i] 

            if i == 0:
                wrcpo_dot[i] = wrcpo[i]
                wlcpo_dot[i] = wlcpo[i]
                xpo[0] = x0
                ypo[0] = y0
                thetapo[0] =theta0
            else:
                wrcpo_dot[i] = wrcpo[i] - wrcpo[i-1]
                wlcpo_dot[i] = wlcpo[i] - wlcpo[i-1]

                xpo[i] += xpo[i-1] + vxpo[i-1]
                ypo[i] += ypo[i-1] + vypo[i-1] 
                thetapo[i] = thetapo[i-1] + omegapo[i]

            vxpo[i] = 1/2*(np.cos(thetapo[i])*wlpo[i]+ np.cos(thetapo[i])*wrpo[i])
            vypo[i] = 1/2*(np.sin(thetapo[i])*wlpo[i]+ np.sin(thetapo[i])*wrpo[i])

def anima(i):
    car.set_data(x[0:i],y[0:i])
    car_main.set_data(x[i],y[i])
    car_com.set_data(xp[0:i],yp[0:i])
    car_com_main.set_data(xp[i],yp[i])
    if online == True:
        car_com_online.set_data(xpo[0:i],ypo[0:i])
        car_com_online_main.set_data(xpo[i],ypo[i])
        return  car, car_com, car_com_online, car_main, car_com_main, car_com_online_main
    else:
        return  car, car_main, car_com, car_com_main, 


if __name__ == "__main__":
    # run simulation
    sta = time.time()

    # multithread on/off
    if multithread == True:
        thread = []
        thread.append(threading.Thread(target = without_compensation))
        thread.append(threading.Thread(target = with_compensation))
        thread.append(threading.Thread(target = with_compensation_online))
        for i in thread:
            i.start()
        for i in thread:
            i.join()
    else:
        without_compensation()
        with_compensation()
        with_compensation_online()
    finish = time.time()
    print("time = ",finish-sta)

    N = time_step
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.gca()
    # ra != 0 -> material changes
    if ra != 1:
        reference_concrete, = ax.plot(xr[:int(N/3)], yr[:int(N/3)], color='blue', linestyle='-', linewidth=2)
        reference_soil, = ax.plot(xr[int(N/3):N], yr[int(N/3):N], color='blue', linestyle='-.', linewidth=2)
    else:
        reference, = ax.plot(xr, yr, color='blue', linestyle='-', linewidth=2)

    car, = ax.plot([], [], color='orange',linestyle='--', linewidth=1)
    car_main, = ax.plot([], [], color='orange', marker='o', markersize=10, markeredgecolor='red', linestyle='')
    car_com, = ax.plot([], [], color='red',linestyle='--', linewidth=1)
    car_com_main, = ax.plot([], [], color='red', marker='o', markersize=10, markeredgecolor='red', linestyle='')
    if online == True:
        car_com_online, = ax.plot([], [], color='green',linestyle='--', linewidth=1)
        car_com_online_main, = ax.plot([], [], color='green', marker='o', markersize=10, markeredgecolor='red', linestyle='')
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ###　plot the simulation result
    ani = animation.FuncAnimation(fig=fig, func=anima, frames=N, interval=0.1, blit=True, repeat=False)
    plt.xlim(-0.5,2)
    plt.ylim(-0.5,2.5)
    plt.title("Tracking Result")
    if online == True:
        if ra != 1:
            plt.legend([reference_concrete, reference_soil, car, car_com, car_com_online,], ['Desired trajectory concrete', 'Desired trajectory soil', 'Trajectory without slip com', 'Trajectory slip com offline' , 'Trajectory slip com online' ], loc='upper left')
            name = 'Tracking compare ra online.gif'
        else:
            plt.legend([reference, car, car_com, car_com_online,], ['Desired trajectory concrete', 'Trajectory without slip com', 'Trajectory slip com offline', 'Trajectory slip com online' ], loc='upper left')
            name = 'Tracking compare online.gif'
    else:
        if ra != 1:
            plt.legend([reference_concrete, reference_soil, car, car_com, ], ['Desired trajectory concrete', 'Desired trajectory soil', 'Trajectory without slip com', 'Trajectory slip com offline' ], loc='upper left')
            name = 'Tracking compare ra.gif'
        else:
            plt.legend([reference, car, car_com, ], ['Desired trajectory concrete', 'Trajectory without slip com', 'Trajectory slip com offline'], loc='upper left')    
            name = 'Tracking compare.gif'
    # save gif
    # ani.save(name, writer='imagemagick', fps=50)
    plt.show()
