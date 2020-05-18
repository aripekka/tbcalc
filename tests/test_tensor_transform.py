# -*- coding: utf-8 -*-
"""
Tests for the tensor transform functions. Run with pytest.

Created on Sat May 9 00:09:00 2020

@author: aripekka
"""

import sys
import os.path
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(__file__),'..'))

from tbcalc.transverse_deformation import * 
from tbcalc import cartesian_tensors_to_cylindrical

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

    RR = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y,X)

    stress, strain, P_imp = isotropic_circular(Rx, Ry, L, thickness, nu, E)

    stress_cyl = cartesian_tensors_to_cylindrical(stress)
    strain_cyl = cartesian_tensors_to_cylindrical(strain)


    stress_cyl_ref = {}
    stress_cyl_ref['rr'] = E/(16*R**2)*(L**2/4-RR**2)+stress['xx'](X,Y)*0
    stress_cyl_ref['phiphi'] = E/(16*R**2)*(L**2/4-3*RR**2)+stress['xx'](X,Y)*0
    stress_cyl_ref['rphi'] = stress['xx'](X,Y)*0
    stress_cyl_ref['phir'] = stress['xx'](X,Y)*0

    strain_cyl_ref = {}
    strain_cyl_ref['rr'] = 1/(16*R**2)*((1-nu)*L**2/4-(1-3*nu)*RR**2)+stress['xx'](X,Y)*0
    strain_cyl_ref['phiphi'] = 1/(16*R**2)*((1-nu)*L**2/4-(3-nu)*RR**2)+stress['xx'](X,Y)*0
    strain_cyl_ref['rphi'] = stress['xx'](X,Y)*0
    strain_cyl_ref['phir'] = stress['xx'](X,Y)*0
    strain_cyl_ref['zphi'] = stress['xx'](X,Y)*0
    strain_cyl_ref['phiz'] = stress['xx'](X,Y)*0
    strain_cyl_ref['rz'] = stress['xx'](X,Y)*0
    strain_cyl_ref['zr'] = stress['xx'](X,Y)*0
    strain_cyl_ref['zz'] = nu/(4*R**2)*(RR**2-L**2/8)+stress['xx'](X,Y)*0

    meps = np.finfo(np.float).eps #m
    
    for i in ['r','phi']:
        for j in ['r','phi']:
            assert np.all(np.logical_or(np.abs(stress_cyl_ref[i+j] - stress_cyl[i+j](RR,PHI)) < meps,
                          np.logical_and(np.isnan(stress_cyl_ref[i+j]), np.isnan(stress_cyl[i+j](RR,PHI)))))

    for i in ['r','phi','z']:
        for j in ['r','phi','z']:
            assert np.all(np.logical_or(np.abs(strain_cyl_ref[i+j] - strain_cyl[i+j](RR,PHI)) < meps,
                          np.logical_and(np.isnan(strain_cyl_ref[i+j]), np.isnan(strain_cyl[i+j](RR,PHI)))))