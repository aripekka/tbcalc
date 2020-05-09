# -*- coding: utf-8 -*-
"""
Tests for the transverse deformation functions. Run with pytest.

Created on Sat May 9 00:09:00 2020

@author: aripekka
"""

import sys
import os.path
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(__file__),'..'))

from tbcalc.transverse_deformation import * 

def test_isotropic_circular():

    #Calculate the reference stresses and strains as implemented in the 
    #deprecated sbcalc package

    E = 165
    nu = 0.22

    thickness = 0.1

    Rx = 1000.0
    Ry = 500.0

    R = np.sqrt(Rx*Ry)
 
    L = 100.0   
 
    x=np.linspace(-L/2,L/2,150)
    X,Y=np.meshgrid(x,x)

    stress = {}
    strain = {}

    stress['xx'] = -E/(16*R**2)*(X**2 + 3*Y**2 -L**2/4)
    stress['yy'] = -E/(16*R**2)*(3*X**2 + Y**2 -L**2/4)
    stress['xy'] = E/(8*R**2)*X*Y
    stress['yx'] = stress['xy']

    strain['xx'] = ((1-nu)*L**2/4-(1-3*nu)*X**2-(3-nu)*Y**2)/(16*R**2)
    strain['yy'] = ((1-nu)*L**2/4-(1-3*nu)*Y**2-(3-nu)*X**2)/(16*R**2)
    strain['xy'] = (1+nu)/(8*R**2)*X*Y
    strain['yx'] = strain['xy']

    strain['zz'] = nu/(4*R**2)*(X**2+Y**2-L**2/8)

    #missing zero strains from sbcalc
    strain['xz'] = X*0
    strain['zx'] = X*0
    strain['yz'] = X*0
    strain['zy'] = X*0

    for k in stress:
        stress[k][X**2+Y**2 > L**2/4] = np.nan
    for k in strain:
        strain[k][X**2+Y**2 > L**2/4] = np.nan

    #add int indexing
    int2char_ind = ['','x','y','z']

    for i in range(1,3):
        for j in range(1,3):
            stress[i*10+j] = stress[int2char_ind[i]+int2char_ind[j]]

    for i in range(1,4):
        for j in range(1,4):
            strain[i*10+j] = strain[int2char_ind[i]+int2char_ind[j]]

    #COMPARE THE REFERENCE TO THE IMPLEMENTATION
    stress_imp, strain_imp, P_imp = isotropic_circular(Rx, Ry, L, thickness, nu, E)
    
    meps = np.finfo(np.float).eps #machine epsilon

    for i in range(1,3):
        for j in range(1,3):
            num_ind = i*10+j
            str_ind = int2char_ind[i]+int2char_ind[j]
            
            assert np.all(np.logical_or(np.abs(stress[num_ind] - stress_imp[num_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(stress[num_ind]), np.isnan(stress_imp[num_ind](X,Y)))))
            assert np.all(np.logical_or(np.abs(stress[str_ind] - stress_imp[str_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(stress[str_ind]), np.isnan(stress_imp[str_ind](X,Y)))))

    for i in range(1,4):
        for j in range(1,4):
            num_ind = i*10+j
            str_ind = int2char_ind[i]+int2char_ind[j]
            
            assert np.all(np.logical_or(np.abs(strain[num_ind] - strain_imp[num_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(strain[num_ind]), np.isnan(strain_imp[num_ind](X,Y)))))
            assert np.all(np.logical_or(np.abs(strain[str_ind] - strain_imp[str_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(strain[str_ind]), np.isnan(strain_imp[str_ind](X,Y)))))

    #check the contact force
    P = -thickness*(stress['xx']/Rx+stress['yy']/Ry)
    
    assert np.all(np.logical_or(np.abs(P - P_imp(X,Y)) < meps,
                                np.logical_and(np.isnan(P), np.isnan(P_imp(X,Y)))))
    