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

def isotropic_circular(Rx,Ry,L,nu,E):
    '''
    Returns the functions to calculate the stress and strain tensor field 
    components as a function of position on the crystal wafer for an isotropic
    toroidally bent crystal analyser.

    Parameters
    ----------
    Rx : float
        Meridional bending radius. The units have to be same as for Ry and L.
    Ry : float
        Sagittal bending radius. The units have to be same as for Rx and L.
    L : float
        Diameter of the wafer. The units have to be same as for Rx and Ry.
    nu : float
        Poisson's ratio.
    E : float
        Young's modulus. Units determine the units of the returned stress 
        tensor components.

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

    '''

    #Define stress functions
    stress = {}
    
    stress['xx'] = lambda x,y : E/(16*Rx*Ry)*(L**2/4 - x**2 - 3*y**2)
    stress['yy'] = lambda x,y : E/(16*Rx*Ry)*(L**2/4 - y**2 - 3*x**2)
    stress['xy'] = lambda x,y : E/(8*Rx*Ry)*x*y
    stress['yx'] = stress['xy']

    #Add alternative indexing
    stress[11] = stress['xx']
    stress[22] = stress['yy']
    stress[12] = stress['xy']    
    stress[21] = stress['yx']

    #Define strain functions
    strain = {}    
    strain['xx'] = lambda x,y : 1/(16*Rx*Ry)*((1 - nu)*L**2/4 -(1 - 3*nu)*x**2 -(3 - nu)*y**2)
    strain['yy'] = lambda x,y : 1/(16*Rx*Ry)*((1 - nu)*L**2/4 -(1 - 3*nu)*y**2 -(3 - nu)*x**2)
    strain['xy'] = lambda x,y : (1 + nu)/(8*Rx*Ry)*x*y
    strain['yx'] = strain['xy']

    strain['xz'] = lambda x,y : np.zeros(np.array(x).shape)
    strain['yz'] = lambda x,y : np.zeros(np.array(x).shape)
    strain['zx'] = strain['xz']
    strain['zy'] = strain['yz']

    strain['zz'] = nu/(4*Rx*Ry)*(x**2 + y**2 - L**2/8)
    
    strain[11] = strain['xx']
    strain[22] = strain['yy']
    strain[12] = strain['xy']    
    strain[21] = strain['yx']

    strain[13] = strain['xz']    
    strain[31] = strain['zx']
    strain[23] = strain['yz']    
    strain[32] = strain['zy']
    
    strain[33] = strain['zz']
    
    return stress, strain