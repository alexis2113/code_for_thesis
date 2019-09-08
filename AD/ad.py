
import autograd as ad
import autograd.numpy as np
import numpy as onp
import pandas as pd
import pickle
from autograd import value_and_grad
from scipy.optimize import root
from autograd.scipy.stats import norm
import types


betas = pd.read_pickle("betas.pkl")
beta1 = betas[0, :]
beta2 = betas[1, :]
target = pd.read_pickle("implied_volatility.pkl")
z1 = np.random.rand(120)
z2 = np.random.rand(120)
z = np.random.rand(120)
dt=0.01
n = norm.pdf
N = norm.cdf

def G(a, b, c, beta1, beta2, z1, z2, T):
    g=(a+b*T)*np.exp(-c*T)*(beta1*z1+beta2*z2)   
    return g

def V(v0,k, v_mean,vov, wt):
    v=v0+k*(v_mean-np.sqrt(abs(v0)))*dt+vov*np.sqrt(abs(v0)*dt)*wt
    
    return v      
# calculate forward rate given parameters

def f_rate(F0,v0,a, b, c, rho, k, theta, ep, betas, zs, z1, z2, tj):
    #instantaous volatility
    rebonato_vol = (a, b, c, betas[0, :], betas[1, :], z1, z2, tj)
    # brownian motion Wt
    wt = (rho / np.sqrt(2)) * (z1 + z2) + np.sqrt(1 - rho ** 2) * zs
    # variance process
    var = V(k, theta, v0, ep, wt)
    # forward rate at time t in period T[j]
    frate = F(F0, rebonato_vol, var, tj)
    return frate

# for a specific combination of S, T, K
v0 = 1.0
F0 = pd.read_pickle("start_rate.pkl")

z = np.random.rand(120)

def f(params):
    a, b, c, theta, kappa, epsilon, rho = params
    estimated = []
    for j in range(30):
        fi = f_rate(F0[j],v0,a, b, c, rho, kappa, theta, epsilon, betas[j], z, z1, z2, j)
        estimated.append(fi)
    diff = target - estimated
    return np.dot(diff)
   
iters = 100
## dy is a function that will return
# 1. the difference between market value and estimated value
# 2. the gradients w.r.t each parameters 
dy = value_and_grad(f)
init_guess = np.array(0.15, 0.015, 0.015, 0.015, 0.015, 0.015, 0.5)
# begin optimization
sol = root(dy, init_guess, jac=True, method='lm',options={"maxiter":iters})
