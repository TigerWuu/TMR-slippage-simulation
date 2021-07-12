import numpy as np

""" 
-------------------------
Author : Jing-Shiang Wuu
Date : 2021/7/12
Institution : National Taiwan University 
Department : Bio-Mechatronics Engineering
Status : Senior
-------------------------

Description:
    Generate the trajectory 

"""
class Trajectory():
    def __init__(self, name, times):
        self.name = name
        self.times = times

    def generator(self):
        t = np.arange(self.times)
        if self.name == "y=x^2":
            xr = 0.01*t 
            yr = (0.01*t) ** 2
            vxr = 0.01*np.ones(self.times)
            vyr = 0.02*(0.01*t)
            vr = (vxr**2 + vyr **2)**0.5
            thetar = np.arctan2(vyr,vxr)
            omegar = 0.02/(0.0004*t**2+1)
        elif self.name == "circle":
            xr = 1.6*np.sin(0.01*t) 
            yr = -1.6*np.cos(0.01*t)+1.6
            vxr = 0.01*1.6*np.cos(0.01*t)
            vyr = 0.01*1.6*np.sin(0.01*t)
            vr = (vxr**2 + vyr **2)**0.5
            thetar = np.arctan2(vyr,vxr)
            omegar = 0.01*np.ones(self.times)
        elif self.name == "y=0.5sin(3x)+x":
            xr = 0.01*t 
            yr = 0.5*np.sin(3*xr) + xr
            vxr = 0.01*np.ones(self.times)
            vyr = 0.015*np.cos(3*xr)+0.01
            vr = (vxr**2 + vyr **2)**0.5
            thetar = np.arctan2(vyr,vxr)
            omegar = -0.045*np.sin(3*xr)/(2.25*(np.cos(3*xr))**2+3*np.cos(3*xr)+2)
        
        return t,xr,yr,vxr,vyr,vr,thetar,omegar
