# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:42:51 2017

@author: HShu
"""

import numpy as np
import dcca

Y_1 = np.loadtxt('Y_1.txt')
Y_2 = np.loadtxt('Y_2.txt')

#Y_k =  X_k + E_k = C_k + D_k + E_k for k=1,2
#where C_1 and C_2 share the same latent factors, but D_1 and D_2 have uncorrelated latent factors.

r_1, r_2, r_12 = 3, 5, 1 # r_k=rank{cov(x_k)}, r_12=rank{cov(x_1,x_2)}; Or see the definition of r_1, r_2 and r_12 in the JASA paper

#Use the true r_1, r_2 and r_12
X_1_hat, X_2_hat, C_1_hat, C_2_hat, D_1_hat, D_2_hat, r_1_hat, r_2_hat, r_12_hat, ccor_hat, theta_hat= dcca.dCCA(Y_1, Y_2, r_1=r_1, r_2=r_2, r_12=r_12)

#Estimate r_1 and r_2 by 'ED', and r_12 by 'MDL-IC'
X_1_hat, X_2_hat, C_1_hat, C_2_hat, D_1_hat, D_2_hat, r_1_hat, r_2_hat, r_12_hat, ccor_hat,theta_hat  = dcca.dCCA(Y_1, Y_2, method='ED')

#Estimate r_1 and r_2 by 'GE', and r_12 by 'MDL-IC'
X_1_hat, X_2_hat, C_1_hat, C_2_hat, D_1_hat, D_2_hat, r_1_hat, r_2_hat, r_12_hat, ccor_hat,theta_hat  = dcca.dCCA(Y_1, Y_2, method='GE')

'''
ccor_hat: the estimated nonzero canonical correlations of x_1 and x_2
theta_hat: arccos(ccor_hat)*180/pi


'ED' method: Onatski, A. (2010), “Determining the number of factors from empirical distribution of eigenval- ues,” The Review of Economics and Statistics, 92, 1004–1016.
'GE' method: Ahn, S. C. and Horenstein, A. R. (2013), “Eigenvalue ratio test for the number of factors,” Econo- metrica, 81, 1203–1227.
'MDL-IC' method: Song, Y., Schreier, P. J., Ram ́ırez, D., and Hasija, T. (2016), “Canonical correlation analysis of high-dimensional data with very small sample support,” Signal Processing, 128, 449–458. 
'''            
