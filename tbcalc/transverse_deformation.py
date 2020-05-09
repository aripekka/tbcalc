# -*- coding: utf-8 -*-
"""
This file contains the functions defining the transverse stress and strain
fields for toroidally bent crystal analysers. For details see: 
Honkanen and Huotari (2020), "General procedure for calculating the elastic 
deformation and X-ray diffraction properties of toroidally and spherically 
bent crystal wafers", In prep.

Created on Fri May  8 14:51:31 2020

@author: aripekka
"""

from __future__ import division, print_function
import numpy as np

def isotropic_circular(Rx,Ry,L,thickness,nu,E):
    '''
    Returns the functions to calculate the stress and strain tensor field 
    components as a function of position on the crystal wafer for an isotropic
    toroidally bent crystal analyser.

    Parameters
    ----------
    Rx : float
        Meridional bending radius
    Ry : float
        Sagittal bending radiu
    L : float
        Diameter of the wafer
    thickness : float
        Thickness of the wafer
    nu : float
        Poisson's ratio
    E : float
        Young's modulus. Units determine the units of the returned stress 
        tensor components.

    For sensible output, the physical units of Rx, Ry, L, and thickness have to
    be the same.

    Returns
    -------
    stress : dict
        Functions returning the transverse stress tensor components as a 
        function of position on the crystal wafer surface. Can be indexed either 
        with x,y or 1,2. For example, sigma_xy at x = X and y = Y is given by

        stress['xy'](X, Y)
        OR 
        stress[12](X, Y)

        Functions return nan for coordinates outside the wafer.

        Units of the position are same as for inputs Rx, Ry, and L, and the 
        unit of stress is that of E.

    strain : dict
        Functions returning the strain tensor components due to the transverse
        stress as a function of position on the crystal wafer surface. Can be 
        indexed either with x,y,z or 1,2,3. For example, epsilon_zz at x = X 
        and y = Y is given by

        strain['zz'](X, Y)
        OR 
        strain[33](X, Y)

        Functions return nan for coordinates outside the wafer.

        Units of the position are same as for inputs Rx, Ry, and L.

    contact_force : function
        Contact force between the wafer and the substrate per unit area as a
        function of position i.e. contact_force(X, Y).

    '''

    #Define stress functions
    stress = {}
    
    def sigma_xx(x,y):
        stress = E/(16*Rx*Ry)*(L**2/4 - x**2 - 3*y**2)
        stress[x**2 + y**2 > L**2/4] = np.nan
        return stress

    def sigma_yy(x,y):
        stress = E/(16*Rx*Ry)*(L**2/4 - y**2 - 3*x**2)
        stress[x**2 + y**2 > L**2/4] = np.nan
        return stress

    def sigma_xy(x,y):
        stress = E/(8*Rx*Ry)*x*y
        stress[x**2 + y**2 > L**2/4] = np.nan
        return stress
        
    stress['xx'] = sigma_xx
    stress['yy'] = sigma_yy
    stress['xy'] = sigma_xy
    stress['yx'] = sigma_xy

    #Add alternative indexing
    stress[11] = stress['xx']
    stress[22] = stress['yy']
    stress[12] = stress['xy']    
    stress[21] = stress['yx']

    #Define strain functions
    strain = {}    

    def epsilon_xx(x,y):
        strain = 1/(16*Rx*Ry)*((1 - nu)*L**2/4 -(1 - 3*nu)*x**2 -(3 - nu)*y**2)
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_yy(x,y):
        strain = 1/(16*Rx*Ry)*((1 - nu)*L**2/4 -(1 - 3*nu)*y**2 -(3 - nu)*x**2)
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_xy(x,y):
        strain = (1 + nu)/(8*Rx*Ry)*x*y
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_xz(x,y):
        strain = np.zeros(np.array(x).shape)
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_yz(x,y):
        strain = np.zeros(np.array(x).shape)
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_zz(x,y):
        strain = nu/(4*Rx*Ry)*(x**2 + y**2 - L**2/8)
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain


    strain['xx'] = epsilon_xx
    strain['yy'] = epsilon_yy
    strain['xy'] = epsilon_xy 
    strain['yx'] = epsilon_xy

    strain['xz'] = epsilon_xz
    strain['zx'] = epsilon_xz
    strain['yz'] = epsilon_yz
    strain['zy'] = epsilon_yz

    strain['zz'] = epsilon_zz
    
    #Add alternative indexing        
    strain[11] = strain['xx']
    strain[22] = strain['yy']
    strain[12] = strain['xy']    
    strain[21] = strain['yx']

    strain[13] = strain['xz']    
    strain[31] = strain['zx']
    strain[23] = strain['yz']    
    strain[32] = strain['zy']
    
    strain[33] = strain['zz']
    
    #Calculate the contact force
    def contact_force(x, y):
        P = E*thickness/(16*Rx**2*Ry**2)*((3*Rx + Ry)*x**2 + (Rx + 3*Ry)*y**2 - (Rx + Ry)*L**2/4)
        P[x**2 + y**2 > L**2/4] = np.nan
        return P
    
    return stress, strain, contact_force

