import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.animation as animation
import csv

time_step = 200
t = np.arange(time_step) 

s1 = np.zeros(time_step)
s2 = np.zeros(time_step)
L = 20

b = 1
a = 0.01
### reference trajectory
# xr = -1.6*np.sin(b*(a*t)**2) 
# yr = -1.6*np.cos(b*(a*t)**2)+1.6
# vxr = -3.2*b*a**2*t*np.cos(b*a**2*t**2)
# vyr = 3.2*b*a**2*t*np.sin(b*a**2*t**2)
# vr = (vxr**2 + vyr **2)**0.5
# thetar = np.arctan2(vyr,vxr)
# omegar = -2*b*a**2*t

xr = 1.6*np.sin(b*(a*t)) 
yr = -1.6*np.cos(b*(a*t))+1.6
vxr = 1.6*b*a*np.cos(b*a*t)
vyr = 1.6*b*a*np.sin(b*a*t)
vr = (vxr**2 + vyr **2)**0.5
thetar = np.arctan2(vyr,vxr)
omegar = b*a*np.ones(time_step)


with open('theta_reference.csv','ab') as f:
    np.savetxt(f, thetar, delimiter="")
with open('x_reference.csv','ab') as f:
    np.savetxt(f, xr, delimiter="")
with open('y_reference.csv','ab') as f:
    np.savetxt(f, yr, delimiter="")


### real trajecotory
x = np.zeros(time_step)
y = np.zeros(time_step)
vx = np.zeros(time_step)
vy = np.zeros(time_step)
v = np.zeros(time_step) 
theta = np.zeros(time_step)
omega = np.zeros(time_step)
wl = np.zeros(time_step)
wr = np.zeros(time_step)

### real trajectory after slippage compensation 

wlp = np.zeros(time_step)
wrp = np.zeros(time_step)

s1p = np.zeros(time_step)
s2p = np.zeros(time_step)

xp = np.zeros(time_step)
yp = np.zeros(time_step)
vxp = np.zeros(time_step)
vyp = np.zeros(time_step)
thetap = np.zeros(time_step)
omegap = np.zeros(time_step)

### controller output
vc = np.zeros(time_step)
omegac = np.zeros(time_step)
wrr = np.zeros(time_step)
wlr = np.zeros(time_step)
wrr_dot = np.zeros(time_step)
wlr_dot = np.zeros(time_step)

### error & error gain
e1 = np.zeros(time_step)
e2 = np.zeros(time_step)
e3 = np.zeros(time_step)

k1 = 0
k2 = 0
k3 = 0

### initial condition
# x0 = 0
# y0 = 0
# theta0 = 0

x0 = 0.1
y0 = -0.1
theta0 = 1/4*np.pi

### slippage distubance
ra = 1.3

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
    vc[i] = vr[i]*np.cos(e3[i]) + k1*e1[i]
    omegac[i] = omegar[i] + k2*np.sign(vr[i])*e2[i] + k3*e3[i]

    ###　inverse kinematic
    wlr[i] = vc[i]*(1/(1-s1[i])) - omegac[i]*L/(2*(1-s1[i]))
    wrr[i] = vc[i]*(1/(1-s2[i])) + omegac[i]*L/(2*(1-s2[i]))
    if i == 0:
        wlr_dot[i] = wlr[i]
        wrr_dot[i] = wrr[i]
    else:
        wlr_dot[i] = wlr[i] - wlr[i-1]
        wrr_dot[i] = wrr[i] - wrr[i-1]

    # slippage ratio empirical equation
    alpha_b = -0.15
    beta_b = -0.63
    alpha_s = 0.07
    beta_s = -0.68

    if wlr[i] >= wrr[i]:
        R =  0.5*L*(wlr[i]+wrr[i])/(wlr[i]-wrr[i])
        alphal = alpha_b
        betal = beta_b
        alphar = alpha_s
        betar = beta_s
    else:
        R =  0.5*L*(wlr[i]+wrr[i])/(wrr[i]-wlr[i])
        alphal = alpha_s
        betal = beta_s
        alphar = alpha_b
        betar = beta_b
    if R>0:
        if i < time_step/3:
            s1[i] = alphal*np.exp(betal*R)
            s2[i] = alphar*np.exp(betar*R)
        else:
            s1[i] = ra*alphal*np.exp(betal*R)
            s2[i] = ra*alphar*np.exp(betar*R)
    else:
        s1[i] = 0
        s2[i] = 0


    ###　forward kinematic
    wl[i] = wlr[i]*(1-s1[i])
    wr[i] = wrr[i]*(1-s2[i])
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
    v[i] = (vx[i]**2 + vy[i]**2)**0.5

with open('theta_without_com.csv','ab') as f:
    np.savetxt(f, theta, delimiter="")
with open('x_without_com.csv','ab') as f:
    np.savetxt(f, x, delimiter="")
with open('y_without_com.csv','ab') as f:
    np.savetxt(f, y, delimiter="")
### load the Network
slippage = keras.models.load_model("slippage_predict")


### with slippage compensation
for i in range(time_step):
    if i == 0:
        s1p[i] = 0
        s2p[i] = 0
        e1[i] = xr[i] - xp[i]
        e2[i] = yr[i] - yp[i]
        e3[i] = thetar[i] - thetap[i]
    else:
        test = [[wrr[i-1],wlr[i-1],wrr_dot[i-1],wlr_dot[i-1],wrp[i-1],wlp[i-1]]]
        sp = slippage.predict(test)
        s1p[i] = sp[0][0]
        s2p[i] = sp[0][1]
        e1[i] = xr[i] - xp[i-1]
        e2[i] = yr[i] - yp[i-1]
        e3[i] = thetar[i] - thetap[i-1]

    k1 = 2 * (omegar[i]**2 + 14*vr[i]**2)**0.5
    k2 = 14*np.abs(vr[i])
    k3 = k1

    ### state feedback & feedforward controller
    vc[i] = vr[i]*np.cos(e3[i]) + k1*e1[i]
    omegac[i] = omegar[i] + k2*np.sign(vr[i])*e2[i] + k3*e3[i]

    ###　inverse kinematic
    wlr[i] = vc[i]*(1/(1-s1p[i])) - omegac[i]*L/(2*(1-s1p[i]))
    wrr[i] = vc[i]*(1/(1-s2p[i])) + omegac[i]*L/(2*(1-s2p[i]))

    # slippage ratio empirical equation

    alpha_b = -0.15
    beta_b = -0.63
    alpha_s = 0.07
    beta_s = -0.68

    if wlr[i] >= wrr[i]:
        R =  0.5*L*(wlr[i]+wrr[i])/(wlr[i]-wrr[i])
        alphal = alpha_b
        betal = beta_b
        alphar = alpha_s
        betar = beta_s
    else:
        R =  0.5*L*(wlr[i]+wrr[i])/(wrr[i]-wlr[i])
        alphal = alpha_s
        betal = beta_s
        alphar = alpha_b
        betar = beta_b
    if R>0:
        if i < time_step/3:
            s1[i] = alphal*np.exp(betal*R)
            s2[i] = alphar*np.exp(betar*R)
        else:
            s1[i] = ra*alphal*np.exp(betal*R)
            s2[i] = ra*alphar*np.exp(betar*R)
    else:
        s1[i] = 0
        s2[i] = 0

    ### forward kinematic
    wlp[i] = wlr[i]*(1-s1[i])
    wrp[i] = wrr[i]*(1-s2[i])
    omegap[i] = 1/L*wrp[i] - 1/L*wlp[i] 

    if i == 0:
        wrr_dot[i] = wrr[i]
        wlr_dot[i] = wlr[i]
        xp[0] = x0
        yp[0] = y0
        thetap[0] =theta0
    else:
        wrr_dot[i] = wrr[i] - wrr[i-1]
        wlr_dot[i] = wlr[i] - wlr[i-1]

        xp[i] += xp[i-1] + vxp[i-1]
        yp[i] += yp[i-1] + vyp[i-1] 
        thetap[i] = thetap[i-1] + omegap[i]

    vxp[i] = 1/2*(np.cos(thetap[i])*wlp[i]+ np.cos(thetap[i])*wrp[i])
    vyp[i] = 1/2*(np.sin(thetap[i])*wlp[i]+ np.sin(thetap[i])*wrp[i])
with open('slippage_com_s1.csv','ab') as f:
    np.savetxt(f, s1, delimiter="")
with open('slippage_com_s2.csv','ab') as f:
    np.savetxt(f, s2, delimiter="")
with open('slippage_com_s1p.csv','ab') as f:
    np.savetxt(f, s1p, delimiter="")
with open('slippage_com_s2p.csv','ab') as f:
    np.savetxt(f, s2p, delimiter="")
with open('theta_with_com.csv','ab') as f:
    np.savetxt(f, thetap, delimiter="")
with open('x_with_com.csv','ab') as f:
    np.savetxt(f, xp, delimiter="")
with open('y_with_com.csv','ab') as f:
    np.savetxt(f, yp, delimiter="")

xpo = np.zeros(time_step)
ypo = np.zeros(time_step)
s1po = np.zeros(time_step)
s2po = np.zeros(time_step)
s1o = np.zeros(time_step)
s2o = np.zeros(time_step)

xpo = np.zeros(time_step)
ypo = np.zeros(time_step)

# with slippage compeensation online
for i in range(time_step):
    if i == 0:
        s1po[i] = 0
        s2po[i] = 0
        e1[i] = xr[i] - xpo[i]
        e2[i] = yr[i] - ypo[i]
        e3[i] = thetar[i] - thetap[i]
    else:
        
        x_online = [[wrr[i-1],wlr[i-1],wrr_dot[i-1],wlr_dot[i-1],wrp[i-1],wlp[i-1]]]
        y_online = [[s1[i-1],s2[i-1]]]
        slippage.fit(x_online , y_online , epochs=3 ,batch_size = 1)
        sp = slippage.predict(x_online) 
        s1po[i] = sp[0][0]
        s2po[i] = sp[0][1]
        e1[i] = xr[i] - xpo[i-1]
        e2[i] = yr[i] - ypo[i-1]
        e3[i] = thetar[i] - thetap[i-1]

    k1 = 2 * (omegar[i]**2 + 14*vr[i]**2)**0.5
    k2 = 14*np.abs(vr[i])
    k3 = k1

    ### state feedback & feedforward controller
    vc[i] = vr[i]*np.cos(e3[i]) + k1*e1[i]
    omegac[i] = omegar[i] + k2*np.sign(vr[i])*e2[i] + k3*e3[i]

    ###　inverse kinematic
    wlr[i] = vc[i]*(1/(1-s1po[i])) - omegac[i]*L/(2*(1-s1po[i]))
    wrr[i] = vc[i]*(1/(1-s2po[i])) + omegac[i]*L/(2*(1-s2po[i]))

    # slippage ratio empirical equation
    alpha_b = -0.15
    beta_b = -0.63
    alpha_s = 0.07
    beta_s = -0.68

    if wlr[i] >= wrr[i]:
        R =  0.5*L*(wlr[i]+wrr[i])/(wlr[i]-wrr[i])
        alphal = alpha_b
        betal = beta_b
        alphar = alpha_s
        betar = beta_s
    else:
        R =  0.5*L*(wlr[i]+wrr[i])/(wrr[i]-wlr[i])
        alphal = alpha_s
        betal = beta_s
        alphar = alpha_b
        betar = beta_b
    if R>0:
        if i < time_step/3:
            s1o[i] = alphal*np.exp(betal*R)
            s2o[i] = alphar*np.exp(betar*R)
        else:
            s1o[i] = ra*alphal*np.exp(betal*R)
            s2o[i] = ra*alphar*np.exp(betar*R)
    else:
        s1[i] = 0
        s2[i] = 0

    ### forward kinematic
    wlp[i] = wlr[i]*(1-s1o[i])
    wrp[i] = wrr[i]*(1-s2o[i])
    omegap[i] = 1/L*wrp[i] - 1/L*wlp[i] 

    if i == 0:
        wrr_dot[i] = wrr[i]
        wlr_dot[i] = wlr[i]
        xpo[0] = x0
        ypo[0] = y0
        thetap[0] =theta0
    else:
        wrr_dot[i] = wrr[i] - wrr[i-1]
        wlr_dot[i] = wlr[i] - wlr[i-1]

        xpo[i] += xpo[i-1] + vxp[i-1]
        ypo[i] += ypo[i-1] + vyp[i-1] 
        thetap[i] = thetap[i-1] + omegap[i]

    vxp[i] = 1/2*(np.cos(thetap[i])*wlp[i]+ np.cos(thetap[i])*wrp[i])
    vyp[i] = 1/2*(np.sin(thetap[i])*wlp[i]+ np.sin(thetap[i])*wrp[i])

with open('slippage_online_s1.csv','ab') as f:
    np.savetxt(f, s1, delimiter="")
with open('slippage_online_s2.csv','ab') as f:
    np.savetxt(f, s2, delimiter="")
with open('slippage_online_s1p.csv','ab') as f:
    np.savetxt(f, s1p, delimiter="")
with open('slippage_online_s2p.csv','ab') as f:
    np.savetxt(f, s2p, delimiter="")
with open('theta_with_online.csv','ab') as f:
    np.savetxt(f, thetap, delimiter="")
with open('x_with_online.csv','ab') as f:
    np.savetxt(f, xpo, delimiter="")
with open('y_with_online.csv','ab') as f:
    np.savetxt(f, ypo, delimiter="")

def anima(i):
    car.set_data(x[0:i],y[0:i])
    car_com.set_data(xp[0:i],yp[0:i])
    car_com_online.set_data(xpo[0:i],ypo[0:i])
    car_main.set_data(x[i],y[i])
    car_com_main.set_data(xp[i],yp[i])
    car_com_online_main.set_data(xpo[i],ypo[i])

    return  car, car_com, car_com_online, car_main, car_com_main, car_com_online_main

if __name__ == "__main__":
    ###　plot the simulation result
    # plt.figure(figsize=(6, 6), dpi=100)
    # for i in range(time_step):
    #     plt.plot(xr[0:i],yr[0:i], color = 'blue')
    #     plt.plot(x[0:i],y[0:i], color = 'orange')
    #     plt.plot(xp[0:i],yp[0:i], color = 'red')
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title("trajectory")
    #     plt.xlim(-0.5,2)
    #     plt.ylim(-0.5,2.5)
    #     if i == time_step-1:
    #         plt.pause(0)
    #     else:
    #         plt.pause(0.0001)
    N = time_step
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.gca()
    reference_concrete, = ax.plot(xr[:66], yr[:66], color='blue', linestyle='-', linewidth=2)
    reference_mud, = ax.plot(xr[66:200], yr[66:200], color='blue', linestyle='-.', linewidth=2)
    
    car, = ax.plot([], [], color='orange',linestyle='--', linewidth=1)
    car_com, = ax.plot([], [], color='red',linestyle='--', linewidth=1)
    car_com_online, = ax.plot([], [], color='green',linestyle='--', linewidth=1)
    car_main, = ax.plot([], [], color='orange', marker='o', markersize=10, markeredgecolor='red', linestyle='')
    car_com_main, = ax.plot([], [], color='red', marker='o', markersize=10, markeredgecolor='red', linestyle='')
    car_com_online_main, = ax.plot([], [], color='green', marker='o', markersize=10, markeredgecolor='red', linestyle='')
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ###　plot the simulation result
    ani = animation.FuncAnimation(fig=fig, func=anima, frames=N, interval=0.1, blit=True, repeat=True)
    plt.xlim(-0.5,2)
    plt.ylim(-0.5,2.5)
    plt.title("Tracking Result Ra")
    plt.legend([reference_concrete, reference_mud, car, car_com, car_com_online,], ['Desired trajectory concrete', 'Desired trajectory mud', 'Trajectory without slip com', 'Trajectory slip com' , 'Trajectory slip com online' ], loc='upper left')
    ani.save('Tracking compare ra.gif', writer='imagemagick', fps=50)
    plt.show()
