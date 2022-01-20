# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:31:15 2022

@author: ARNON ARCHANKUL
"""

import numpy as np

class ABM:
    
    # constructor
    # parameters from the arithmetic brownian motions (ABM) and control costs
    def __init__(self, rho, alpha, sigma, kappa):
        self.rho = rho
        self.alpha = alpha
        self.sigma = sigma
        self.kappa = kappa
        
    # setters
    
    def set_rho(self, x):
        self.rho = x
        
    def set_alpha(self, x):
        self.alpha = x
        
    def set_sigma(self, x):
        self.sigma = x
        
    def set_kappa(self, x):
        self.kappa = x
    
    # roots from solving quadratic equations subjected to the uncontrolled cost function solutions with ambiguity parameter +kappa and -kappa
    # ks = kappa sign {-1,1}
    
    def beta(self, ks):
        return ( -self.alpha - ks*self.kappa*self.sigma + np.sqrt(2*self.rho*np.square(self.sigma) + np.square(self.alpha + ks*self.kappa*self.sigma) ) ) / np.square(self.sigma)
    
    def gamma(self, ks):
        return ( -self.alpha - ks*self.kappa*self.sigma - np.sqrt(2*self.rho*np.square(self.sigma) + np.square(self.alpha + ks*self.kappa*self.sigma) ) ) / np.square(self.sigma)
    
    # A convex increasing function with +kappa or -kappa at nder order derivatives
    def vh(self, x, ks, nder): 
        return np.power(self.beta(ks), nder) * np.exp(x*self.beta(ks)) 
    
    # A convex decreasing function with +kappa or -kappa at nder order derivatives
    def vc(self, x, ks, nder):
        return np.power(self.gamma(ks), nder) * np.exp(x*self.gamma(ks))
    
    
    
    
    
        
    