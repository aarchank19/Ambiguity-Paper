# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:30:24 2022

@author: ARNON ARCHANKUL
"""

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.font_manager as font
import solcost
import sys

alpha = 0.2
kappa = 0.0
sigma = 3.5
rho = .05

l = 0.5
u = 0.5
cc = 1
ch = 1

sc = solcost.SolCost(alpha, kappa, sigma, rho, l, u, cc, ch)



x = np.linspace(-10.0, 0, num = 50)

z = np.linspace(0, 10, num = 50)

def test_l(x):
    return [sc.l + sc.R(y, -1, 1) - ( sc.vc(y, -1, 1)/sc.vc(y, -1, 2) ) * sc.R(y, -1, 2) for y in x]

def test_u(x):
    return [sc.u - sc.R(y, 1, 1) + ( sc.vh(y, 1, 1)/sc.vh(y, 1, 2) ) * sc.R(y, 1, 2) for y in x]
    

# plt.plot(z, test_u(z))

# kp = np.array([0, 0.5, 1])  # kappa
# cf = np.zeros((3, x.size))  # cost funtion
# cr = np.zeros((kp.size, 2, kp.size))    # continuous regions
# for i in range(kp.size) :
#     sc.set_kappa(kp[i])
#     tg = sc.xv()
#     cr[i,0,:] = tg
#     cr[i,1,:] = np.array([ sc.J(tg[0]), sc.J(tg[1]), sc.J(tg[2])])
#     for j in range(cf.shape[1]):
#         cf[i,j] = sc.J(x[j])
    
# plt.plot(x, cf[0,:],'-b', label = '$\kappa=0$')
# plt.plot([cr[0,0,0], cr[0,0,0]], [0, cr[0,1,0]], '--b')
# plt.plot([cr[0,0,1], cr[0,0,1]], [0, cr[0,1,1]], '--b')
# plt.plot([cr[0,0,2], cr[0,0,2]], [0, cr[0,1,2]], '--b')

# plt.plot(x, cf[1,:],'-r', label = '$\kappa=0.5$')
# plt.plot([cr[1,0,0], cr[1,0,0]], [0, cr[1,1,0]], '--r')
# plt.plot([cr[1,0,1], cr[1,0,1]], [0, cr[1,1,1]], '--r')
# plt.plot([cr[1,0,2], cr[1,0,2]], [0, cr[1,1,2]], '--r')

# plt.plot(x, cf[2,:],'-g', label = '$\kappa=1$')
# plt.plot([cr[2,0,0], cr[2,0,0]], [0, cr[2,1,0]], '--g')
# plt.plot([cr[2,0,1], cr[2,0,1]], [0, cr[2,1,1]], '--g')
# plt.plot([cr[2,0,2], cr[2,0,2]], [0, cr[2,1,2]], '--g')
 
# plt.legend()

# plt.show()


p = np.arange(.1, 1.1, .1)

sc.set_kappa(0.0)
tgs = np.zeros((3, p.size))
for k in range(tgs.shape[1]):
    sc.set_sigma(p[k])
    tgs[:,k]= sc.xv()
    

sc.set_kappa(0.5)
tgs2 = np.zeros((3, p.size))
for k in range(tgs.shape[1]):
    sc.set_sigma(p[k])
    tgs2[:,k]= sc.xv()
    
sc.set_kappa(1)
tgs3 = np.zeros((3, p.size))
for k in range(tgs.shape[1]):
    sc.set_sigma(p[k])
    tgs3[:,k]= sc.xv()
        
plt.rcParams["font.family"] = "Times New Roman"

plt.plot(p, tgs[0,:], '-b', label = '$\kappa$ = 0')
plt.plot(p, tgs[1,:], '-b')
plt.plot(p, tgs[2,:], '-b')

plt.plot(p, tgs2[0,:], '-r', label = '$\kappa$ = 0.5')
plt.plot(p, tgs2[1,:], '-r')
plt.plot(p, tgs2[2,:], '-r')

plt.plot(p, tgs3[0,:], '-g', label = '$\kappa$ = 1')
plt.plot(p, tgs3[1,:], '-g')
plt.plot(p, tgs3[2,:], '-g')

plt.xlabel('$\sigma$')
plt.ylabel('$x$')
plt.axhline(y=0, linestyle ='dotted', color = 'black')

plt.legend()

plt.savefig('triggers.png', dpi=300)

plt.show()


