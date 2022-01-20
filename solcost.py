# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:35:32 2022

@author: ARNON ARCHANKUL
"""

import sys
import numpy as np
import scipy.optimize as opt
from abm import ABM

class SolCost(ABM):
    
    #construtor
    def __init__(self, rho, alpha, sigma, kappa, l, u, cc, ch):
        super().__init__(rho, alpha, sigma, kappa)
        self.l = l
        self.u = u
        self.cc = cc
        self.ch = ch
        
    #=======================================================================
    # ks = kappa sign {-1, 1} 
    # nder is a derivative order
    #=======================================================================
    
    #=======================================================================
    # Parameters for an uncontrolled perpetual cost function (R)
    #=======================================================================
    
    def E(self, ks):
        return (self.cc + self.ch)*np.square(self.sigma*self.gamma(ks)) / ((2*np.square(self.rho))*(self.beta(ks) - self.gamma(ks)))
    
    def F(self, ks):
        return (self.cc + self.ch)*np.square(self.sigma*self.beta(ks)) / ((2*np.square(self.rho))*(self.beta(ks) - self.gamma(ks)))
    
    def f(self, x, ks, nder): 
        if nder == 0 : 
            return -self.cc*x/self.rho  - self.cc*(self.alpha + ks*self.kappa*self.sigma)/np.square(self.rho)
        elif nder == 1 :
            return -self.cc/self.rho
        else : 
            return 0
    
    def g(self, x, ks, nder): 
        if nder == 0 : 
            return self.ch*x/self.rho + self.ch*(self.alpha + ks*self.kappa*self.sigma)/np.square(self.rho)
        elif nder == 1 :
            return self.ch/self.rho
        else : 
            return 0
    
    def R(self, x, ks, nder):
        # if nder not in [0,1,2,3] : 
        #     print('The derivative of R is invalid. Please input derivative order in integer')
        #     sys.exit()
        if x<0 : 
            return self.f(x, ks, nder) + self.E(ks)*self.vh(x, ks, nder)
        else : 
            return self.g(x, ks, nder) + self.F(ks)*self.vc(x, ks, nder)
       
    #=======================================================================
    # Parameters for a controlled cost function (J)
    #=======================================================================
    
    def A(self, x, nder):
        ks = -1
        if nder == 0 :
            return ( self.vc(x, ks, 2)*(-self.l - self.R(x, ks, 1)) + self.vc(x, ks, 1)*self.R(x, ks, 2) ) / ( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) )
        if nder == 1 :
            return ( (self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2))*self.R(x, ks, 3) + (self.vc(x, ks, 1)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 1))*self.R(x, ks, 2) + (self.vc(x, ks, 2)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 2))*(-self.l - self.R(x, ks, 1)) ) / np.square( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) ) * self.vc(x, ks, 1)
        
    def B(self, x, nder):
        ks = -1
        if nder == 0 :
            return - ( self.vh(x, ks, 2)*(-self.l - self.R(x, ks, 1)) + self.vh(x, ks, 1)*self.R(x, ks, 2) ) / ( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) )
        if nder == 1 :
            return - ( (self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2))*self.R(x, ks, 3) + (self.vc(x, ks, 1)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 1))*self.R(x, ks, 2) + (self.vc(x, ks, 2)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 2))*(-self.l - self.R(x, ks, 1)) ) / np.square( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) ) * self.vh(x, ks, 1)
    
    def C(self, x, nder):
        ks = 1
        if nder == 0 :
            return ( self.vc(x, ks, 2)*(self.u - self.R(x, ks, 1)) + self.vc(x, ks, 1)*self.R(x, ks, 2) ) / ( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) )
        if nder == 1 :
            return ( (self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2))*self.R(x, ks, 3) + (self.vc(x, ks, 1)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 1))*self.R(x, ks, 2) + (self.vc(x, ks, 2)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 2))*(self.u - self.R(x, ks, 1)) ) / np.square( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) ) * self.vc(x, ks, 1)
        
    def D(self, x, nder):
        ks = 1
        if nder == 0 :
            return - ( self.vh(x, ks, 2)*(self.u - self.R(x, ks, 1)) + self.vh(x, ks, 1)*self.R(x, ks, 2) ) / ( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) )
        if nder == 1 :
            return - ( (self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2))*self.R(x, ks, 3) + (self.vc(x, ks, 1)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 1))*self.R(x, ks, 2) + (self.vc(x, ks, 2)*self.vh(x, ks, 3) - self.vc(x, ks, 3)*self.vh(x, ks, 2))*(self.u - self.R(x, ks, 1)) ) / np.square( self.vc(x, ks, 2)*self.vh(x, ks, 1) - self.vc(x, ks, 1)*self.vh(x, ks, 2) ) * self.vh(x, ks, 1)
   
    
    # finding   xl = lower bound control 
    #           xu = upper bound control and 
    #           xs = ambiguity trigger 
    
    # define a function given by assumptions 1-3 from Proposition 2 to solve xl, xs, xu
    def funct(self, xv): # xv = [xl, xs, xu ]
        #left smooth pasting
        fl = self.A(xv[0], 0)*self.vh(xv[1], -1, 1) + self.B(xv[0], 0)*self.vc(xv[1], -1, 1) + self.R(xv[1], -1, 1)
        
        #value matching
        fm = self.A(xv[0], 0)*self.vh(xv[1], -1, 0) + self.B(xv[0], 0)*self.vc(xv[1], -1, 0) + self.R(xv[1], -1, 0) - ( self.C(xv[2], 0)*self.vh(xv[1], 1, 0) + self.D(xv[2], 0)*self.vc(xv[1], 1, 0) + self.R(xv[1], 1, 0) )
        
        #right smooth pasting
        fr = self.C(xv[2], 0)*self.vh(xv[1], 1, 1) + self.D(xv[2], 0)*self.vc(xv[1], 1, 1) + self.R(xv[1], 1, 1)
    
        return np.array([fl, fm, fr])

    # define a Jacobian matrix for a root-finding problem to get xl, xs, xu (optional)
    def jac(self, xv):
        
        dfl_dxl = self.A(xv[0], 1)*self.vh(xv[1], -1, 1) + self.B(xv[0], 1)*self.vc(xv[1], -1, 1)
        dfl_dxs = self.A(xv[0], 0)*self.vh(xv[1], -1, 2) + self.B(xv[0], 0)*self.vc(xv[1], -1, 2) + self.R(xv[1], -1, 2)
        dfl_dxu = 0.0
        
        dfm_dxl = self.A(xv[0], 1)*self.vh(xv[1], -1, 0) + self.B(xv[0], 1)*self.vc(xv[1], -1, 0) 
        dfm_dxs = self.A(xv[0], 0)*self.vh(xv[1], -1, 1) + self.B(xv[0], 0)*self.vc(xv[1], -1, 1) + self.R(xv[1], -1, 1) - ( self.C(xv[2], 0)*self.vh(xv[1], 1, 1) + self.D(xv[2], 0)*self.vc(xv[1], 1, 1) + self.R(xv[1], 1, 1) )
        dfm_dxu = -( self.C(xv[2], 1)*self.vh(xv[1], 1, 0) + self.D(xv[2], 1)*self.vc(xv[1], 1, 0) )
                    
        dfr_dxl = 0.0
        dfr_dxs = self.C(xv[2], 0)*self.vh(xv[1], 1, 2) + self.D(xv[2], 0)*self.vc(xv[1], 1, 2) + self.R(xv[1], 1, 2)
        dfr_dxu = self.C(xv[2], 1)*self.vh(xv[1], 1, 1) + self.D(xv[2], 1)*self.vc(xv[1], 1, 1)
        
        return np.array( [ [dfl_dxl, dfl_dxs, dfl_dxu], [dfm_dxl, dfm_dxs, dfm_dxu], [dfr_dxl, dfr_dxs, dfr_dxu] ])
    
    def xv(self):
        x0 = np.array([-0.5,0,0.5])
        rt = opt.root(self.funct, x0, jac=self.jac)
        return rt.x
    
    def J(self, x):
        xl = self.xv()[0]
        xs = self.xv()[1]
        xu = self.xv()[2]
        nder = 0
        if x <=  xl:
            ks = -1
            return self.l*(xl - x) + self.R(xl, ks, nder) + self.A(xl, nder)*self.vh(xl, ks, nder) + self.B(xl, nder)*self.vc(xl, ks, nder)
        elif x > xl and x < xs :
            ks = -1
            return self.R(x, ks, nder) + self.A(xl, nder)*self.vh(x, ks, nder) + self.B(xl, nder)*self.vc(x, ks, nder)
        elif x >= xs and x < xu :
            ks = 1
            return self.R(x, ks, nder) + self.C(xu, nder)*self.vh(x, ks, nder) + self.D(xu, nder)*self.vc(x, ks, nder)
        else :
            ks = 1
            return self.u*(x - xu) + self.R(xu, ks, nder) + self.C(xu, nder)*self.vh(xu, ks, nder) + self.D(xu, nder)*self.vc(xu, ks, nder)
    
        