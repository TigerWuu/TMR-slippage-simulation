import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import DNN as nn



### simulation step
time_step = 200

### car width
L = 20

### reference trajectory
# xr = 0.01*t 
# yr = 0.5*np.sin(3*xr) + xr
# vxr = 0.01*np.ones(time_step)
# vyr = 0.015*np.cos(3*xr)+0.01
# vr = (vxr**2 + vyr **2)**0.5
# thetar = np.arctan2(vyr,vxr)
# omegar = -0.045*np.sin(3*xr)/(2.25*(np.cos(3*xr))**2+3*np.cos(3*xr)+2)
t = np.arange(time_step) 
xr = 0.01*t 
yr = (0.01*t) ** 2
vxr = 0.01*np.ones(time_step)
vyr = 0.02*(0.01*t)
vr = (vxr**2 + vyr **2)**0.5
thetar = np.arctan2(vyr,vxr)
omegar = 0.02/(0.0004*t**2+1)

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
s1 = np.zeros(time_step)
s2 = np.zeros(time_step)

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
x0 = 0
y0 = 0
theta0 = 0

for i in range(time_step):
    if thetar[i] < 0:
        thetar[i] += 2*np.pi
    if i == 0:
        e1[i] = xr[i] - x[i]
        e2[i] = yr[i] - y[i]
        e3[i] = thetar[i] - theta[i]
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
    alphal = 0
    betal = 0
    alphar = 0
    betar = 0

    if wlr[i] >= wrr[i]:
        R =  0.5*L*(wlr[i]+wrr[i])/(wlr[i]-wrr[i])
        alphal = -0.15
        betal = -0.63
        alphar = 0.07
        betar = -0.68
    else:
        R =  0.5*L*(wlr[i]+wrr[i])/(wrr[i]-wlr[i])
        alphal = 0.07
        betal = -0.68
        alphar = -0.15
        betar = -0.63
    if R>0:
        s1[i] = alphal*np.exp(betal*R)
        s2[i] = alphar*np.exp(betar*R)
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


### predict the slippage ratio

x_train = np.concatenate((wrr[0:time_step-1].reshape(time_step-1,1),wlr[0:time_step-1].reshape(time_step-1,1),wrr_dot[0:time_step-1].reshape(time_step-1,1),wlr_dot[0:time_step-1].reshape(time_step-1,1),wr[0:time_step-1].reshape(time_step-1,1),wl[0:time_step-1].reshape(time_step-1,1)),axis = 1)
y_train = np.concatenate((s1[1:time_step].reshape(time_step-1,1),s2[1:time_step].reshape(time_step-1,1)),axis = 1)

slippage = keras.models.Sequential([
    keras.Input(shape=6),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(2, activation='linear')
])

mse = keras.losses.MeanSquaredError()
sgd = keras.optimizers.SGD()
slippage.compile(loss=mse , optimizer=sgd , metrics= ["accuracy"])

his = slippage.fit(x_train , y_train , epochs=200 ,batch_size = 32)
# plt.plot(his.history['loss'])
# plt.show()

slippage.save("slippage_predict")


### DNN test
# slippage = nn.DNN([6,10,2])
# slippage.training(x_train , y_train , 200 ,0.1)

# real slippage

for i in range(time_step):
    if thetar[i] < 0:
        thetar[i] += 2*np.pi
    if i == 0:
        s1p[i] = 0
        s2p[i] = 0
        e1[i] = xr[i] - xp[i]
        e2[i] = yr[i] - yp[i]
        e3[i] = thetar[i] - thetap[i]
    else:
        test = [[wrr[i-1],wlr[i-1],wrr_dot[i-1],wlr_dot[i-1],wrp[i-1],wlp[i-1]]]  #
        sp = slippage.predict(test)
        s1p[i] = sp[0][0]
        s2p[i] = sp[0][1]
        # s1p[i] = sp[0]
        # s2p[i] = sp[1]
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
    alphal = 0
    betal = 0
    alphar = 0
    betar = 0
    if wlr[i] >= wrr[i]:
        R =  0.5*L*(wlr[i]+wrr[i])/(wlr[i]-wrr[i])
        alphal = -0.15
        betal = -0.63
        alphar = 0.07
        betar = -0.68
    else:
        R =  0.5*L*(wlr[i]+wrr[i])/(wrr[i]-wlr[i])
        alphal = 0.07
        betal = -0.68
        alphar = -0.15
        betar = -0.63
    if R>0:
        s1[i] = alphal*np.exp(betal*R)
        s2[i] = alphar*np.exp(betar*R)
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


if __name__ == "__main__":
    ###　plot the simulation result
    # plt.plot(vxr, color = 'blue')
    # plt.plot(vx, color = 'orange')
    # plt.plot(vxp, color = 'red')
    # plt.title("velocity x")
    # plt.figure()
    # plt.plot(vyr, color = 'blue')
    # plt.plot(vy, color = 'orange')
    # plt.plot(vyp, color = 'red')
    # plt.title("velocity y")
    # plt.figure()
    plt.plot(xr,yr, color = 'blue')
    plt.plot(x,y, color = 'orange')
    plt.plot(xp,yp, color = 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(['Desired trajectory', 'Trajectory without slip com', 'Trajectory slip com'], loc='upper left')
    plt.title("Trajectory")
    plt.figure()
    # plt.plot(thetar, color = 'blue')
    # plt.plot(theta, color = 'orange')
    # plt.plot(thetap, color = 'red')
    # plt.title("theta")
    # plt.figure()
    plt.plot(s1, color = 'darkblue')
    plt.plot(s2, color = 'cyan')
    plt.plot(s1p, color = 'darkgreen')
    plt.plot(s2p, color = 'lime')
    plt.ylim(-1,1)
    plt.legend(['slippage left', 'slippage right', 'slippage left predict' ,'slippage right predict'], loc='upper left')
    plt.title("slippage ratio")
    plt.figure()
    plt.plot(s1-s1p, color = 'darkblue')
    plt.plot(s2-s2p, color = 'darkgreen')
    plt.ylim(-1,1)
    plt.legend(['slippage left error', 'slippage right error'], loc='upper left')
    plt.title("slippage ratio error")
    plt.show()
