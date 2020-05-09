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

from pyTTE import TTcrystal, Quantity

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

def test_anisotropic_circular_vs_sbcalc():

    #For compliance matrix
    ttx = TTcrystal(crystal='Si',hkl=[9,5,1], thickness=Quantity(0.1,'mm'))

    S = ttx.S.in_units('GPa^-1')

    thickness = 0.1

    Rx = 1000.0
    Ry = 500.0

    R = np.sqrt(Rx*Ry)
 
    L = 100.0   

    #Calculate the stresses and strains as in the now deprecated sbcalc
    x=np.linspace(-L/2,L/2,150)
    X,Y=np.meshgrid(x,x)

    r_squared = X**2+Y**2
    phi = np.arctan2(Y,X)

    stress = {}
    strain = {}

    D = 1/(2*R**2*(3*(S[0,0]+S[1,1])+2*S[0,1]+S[5,5]))

    stress['xx'] = D*(L**2/4-X**2-3*Y**2)
    stress['yy'] = D*(L**2/4-3*X**2-Y**2)
    stress['xy'] = 2*D*X*Y
    stress['yx'] = stress['xy']

    #shorthand notation
    uzzaux1 = (S[2,0]+S[2,1])*L**2/4
    uzzaux2 = 2*(S[2,0]+S[2,1])
    uzzaux3 = np.sqrt((S[2,1]-S[2,0])**2+S[2,5]**2)

    beta = np.arctan2(S[2,5],(S[2,1]-S[2,0]))

    strain['zz'] = D*(uzzaux1 - (uzzaux2+uzzaux3*np.cos(2*phi+beta))*r_squared) #In sbcalc, there's incorrectly sin instead of cos

    for k in stress:
        stress[k][X**2+Y**2 > L**2/4] = np.nan
    for k in strain:
        strain[k][X**2+Y**2 > L**2/4] = np.nan

    stress_imp, strain_imp, P_imp = anisotropic_circular(Rx, Ry, L, thickness, S)
    
    meps = np.finfo(np.float).eps #machine epsilon

    #add int indexing
    int2char_ind = ['','x','y','z']

    for i in range(1,3):
        for j in range(1,3):
            str_ind = int2char_ind[i]+int2char_ind[j]       
            assert np.all(np.logical_or(np.abs(stress[str_ind] - stress_imp[str_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(stress[str_ind]), np.isnan(stress_imp[str_ind](X,Y)))))    

    assert np.all(np.logical_or(np.abs(strain['zz'] - strain_imp['zz'](X,Y)) < meps,
                                np.logical_and(np.isnan(strain['zz']), np.isnan(strain_imp['zz'](X,Y)))))


    #check the contact force
    P = -thickness*(stress['xx']/Rx+stress['yy']/Ry)
    
    assert np.all(np.logical_or(np.abs(P - P_imp(X,Y)) < meps,
                                np.logical_and(np.isnan(P), np.isnan(P_imp(X,Y)))))

def test_anisotropic_circular_vs_isotropic_circular():

    E = 165
    nu = 0.22

    thickness = 0.1

    Rx = 1000.0
    Ry = 500.0

    L = 100.0
    
    S = np.zeros((6,6))

    #The elastic matrix for isotropic crystal
    S[0,0] = 1
    S[1,1] = 1
    S[2,2] = 1

    S[0,1] = -nu
    S[0,2] = -nu
    S[1,2] = -nu
    S[1,0] = -nu
    S[2,0] = -nu
    S[2,1] = -nu

    S[3,3] = 2*(1+nu)
    S[4,4] = 2*(1+nu)
    S[5,5] = 2*(1+nu)

    S = S/E

    stress_iso, strain_iso, P_iso = isotropic_circular(Rx, Ry, L, thickness, nu, E)    
    stress_aniso, strain_aniso, P_aniso = anisotropic_circular(Rx, Ry, L, thickness, S)

    x=np.linspace(-L/2,L/2,150)
    X,Y=np.meshgrid(x,x)

    meps = np.finfo(np.float).eps #machine epsilon

    int2char_ind = ['','x','y','z']
    
    meps = np.finfo(np.float).eps #machine epsilon

    #Check stresses
    for i in range(1,3):
        for j in range(1,3):
            num_ind = i*10+j
            str_ind = int2char_ind[i]+int2char_ind[j]
            
            assert np.all(np.logical_or(np.abs(stress_iso[num_ind](X,Y) - stress_aniso[num_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(stress_iso[num_ind](X,Y)), np.isnan(stress_aniso[num_ind](X,Y)))))
            assert np.all(np.logical_or(np.abs(stress_iso[str_ind](X,Y) - stress_aniso[str_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(stress_iso[str_ind](X,Y)), np.isnan(stress_aniso[str_ind](X,Y)))))

    #Check strains
    for i in range(1,4):
        for j in range(1,4):
            num_ind = i*10+j
            str_ind = int2char_ind[i]+int2char_ind[j]
            
            assert np.all(np.logical_or(np.abs(strain_iso[num_ind](X,Y) - strain_aniso[num_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(strain_iso[num_ind](X,Y)), np.isnan(strain_aniso[num_ind](X,Y)))))
            assert np.all(np.logical_or(np.abs(strain_iso[str_ind](X,Y) - strain_aniso[str_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(strain_iso[str_ind](X,Y)), np.isnan(strain_aniso[str_ind](X,Y)))))

    #Check contact forces
    assert np.all(np.logical_or(np.abs(P_iso(X,Y) - P_aniso(X,Y)) < meps,
                                np.logical_and(np.isnan(P_iso(X,Y)), np.isnan(P_aniso(X,Y)))))
    
def test_isotropic_rectangular():

    #Calculate the reference stresses and strains as implemented in the 
    #deprecated sbcalc package

    E = 165
    nu = 0.22

    thickness = 0.1

    Rx = 1000.0
    Ry = 500.0

    R = np.sqrt(Rx*Ry)
 
    a = 100.0   
    b = 50.0   
    
    x=np.linspace(-a/2,a/2,150)
    X,Y=np.meshgrid(x,x)

    stress = {}
    strain = {}

    g = 8 + 10*((a/b)**2+(b/a)**2) + (1-nu)*((a/b)**2-(b/a)**2)**2

    stress['xx'] = E/(g*R**2) * (a**2/12-X**2 + ((1+nu)/2 + 5*(a/b)**2 + (1-nu)/2*(a/b)**4)*(b**2/12-Y**2))
    stress['yy'] = E/(g*R**2) * (b**2/12-Y**2 + ((1+nu)/2 + 5*(b/a)**2 + (1-nu)/2*(b/a)**4)*(a**2/12-X**2)) #sbcalc has a typo on this line (corrected here)
    stress['xy'] = 2*E/(g*R**2)*X*Y
    stress['yx'] = stress['xy']

    strain['zz'] = nu/(g*R**2) * (((3+nu)/2+5*(b/a)**2+(1-nu)/2*(b/a)**4)*(X**2 - a**2/12)+\
                                  ((3+nu)/2+5*(a/b)**2+(1-nu)/2*(a/b)**4)*(Y**2 - b**2/12))

    for k in stress:
        stress[k][np.abs(X) > a/2] = np.nan
        stress[k][np.abs(Y) > b/2] = np.nan        
    for k in strain:
        strain[k][np.abs(X) > a/2] = np.nan
        strain[k][np.abs(Y) > b/2] = np.nan  

    #add int indexing
    int2char_ind = ['','x','y','z']

    for i in range(1,3):
        for j in range(1,3):
            stress[i*10+j] = stress[int2char_ind[i]+int2char_ind[j]]

    for i in range(3,4):
        for j in range(3,4):
            strain[i*10+j] = strain[int2char_ind[i]+int2char_ind[j]]

    #COMPARE THE REFERENCE TO THE IMPLEMENTATION
    stress_imp, strain_imp, P_imp = isotropic_rectangular(Rx, Ry, a, b, thickness, nu, E)
    
    meps = np.finfo(np.float).eps #machine epsilon
    
    for i in range(1,3):
        for j in range(1,3):

            num_ind = i*10+j
            str_ind = int2char_ind[i]+int2char_ind[j]

            print(str_ind)
            
            assert np.all(np.logical_or(np.abs(stress[num_ind] - stress_imp[num_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(stress[num_ind]), np.isnan(stress_imp[num_ind](X,Y)))))
            assert np.all(np.logical_or(np.abs(stress[str_ind] - stress_imp[str_ind](X,Y)) < meps,
                                        np.logical_and(np.isnan(stress[str_ind]), np.isnan(stress_imp[str_ind](X,Y)))))

    for i in range(3,4):
        for j in range(3,4):
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

    