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
    circular toroidally bent crystal analyser.

    Parameters
    ----------
    Rx : float
        Meridional bending radius
    Ry : float
        Sagittal bending radius
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

def anisotropic_circular(Rx,Ry,L,thickness,S):
    '''
    Returns the functions to calculate the stress and strain tensor field 
    components as a function of position on the crystal wafer for an anisotropic
    circular toroidally bent crystal analyser.

    Parameters
    ----------
    Rx : float
        Meridional bending radius
    Ry : float
        Sagittal bending radius
    L : float
        Diameter of the wafer
    thickness : float
        Thickness of the wafer
    S : 6x6 numpy array
        Compliance matrix in the Voigt notation. The inverse of units determine
        the units of the returned stress tensor components.

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

    S = S.copy()

    #Effective Young's modulus
    E_eff = 8/(3*(S[0,0]+S[1,1]) + 2*S[0,1] + S[5,5])

    #Define stress functions
    stress = {}
    
    def sigma_xx(x,y):
        stress = E_eff/(16*Rx*Ry)*(L**2/4 - x**2 - 3*y**2)
        stress[x**2 + y**2 > L**2/4] = np.nan
        return stress

    def sigma_yy(x,y):
        stress = E_eff/(16*Rx*Ry)*(L**2/4 - y**2 - 3*x**2)
        stress[x**2 + y**2 > L**2/4] = np.nan
        return stress

    def sigma_xy(x,y):
        stress = E_eff/(8*Rx*Ry)*x*y
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
        strain = E_eff/(16*Rx*Ry)*(  (S[0,0] + S[0,1])*L**2/4 
                                   - (S[0,0] + 3*S[0,1])*x**2
                                   - (3*S[0,0] + S[0,1])*y**2 
                                   + 2*S[0,5]*x*y
                                  )
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_yy(x,y):
        strain = E_eff/(16*Rx*Ry)*(  (S[1,0] + S[1,1])*L**2/4 
                                   - (S[1,0] + 3*S[1,1])*x**2
                                   - (3*S[1,0] + S[1,1])*y**2 
                                   + 2*S[1,5]*x*y
                                  )
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_zz(x,y):
        strain = E_eff/(16*Rx*Ry)*(  (S[2,0] + S[2,1])*L**2/4
                                   - (S[2,0] + 3*S[2,1])*x**2
                                   - (3*S[2,0] + S[2,1])*y**2 
                                   + 2*S[2,5]*x*y
                                  )
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_xz(x,y):
        strain = E_eff/(32*Rx*Ry)*(  (S[3,0] + S[3,1])*L**2/4 
                                   - (S[3,0] + 3*S[3,1])*x**2
                                   - (3*S[3,0] + S[3,1])*y**2 
                                   + 2*S[3,5]*x*y
                                  )
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_yz(x,y):
        strain = E_eff/(32*Rx*Ry)*(  (S[4,0] + S[4,1])*L**2/4 
                                   - (S[4,0] + 3*S[4,1])*x**2
                                   - (3*S[4,0] + S[4,1])*y**2 
                                   + 2*S[4,5]*x*y
                                  )
        strain[x**2 + y**2 > L**2/4] = np.nan
        return strain

    def epsilon_xy(x,y):
        strain = E_eff/(32*Rx*Ry)*(  (S[5,0] + S[5,1])*L**2/4 
                                   - (S[5,0] + 3*S[5,1])*x**2
                                   - (3*S[5,0] + S[5,1])*y**2 
                                   + 2*S[5,5]*x*y
                                  )
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
        P = E_eff*thickness/(16*Rx**2*Ry**2)*((3*Rx + Ry)*x**2 + (Rx + 3*Ry)*y**2 - (Rx + Ry)*L**2/4)
        P[x**2 + y**2 > L**2/4] = np.nan
        return P
    
    return stress, strain, contact_force

def isotropic_rectangular(Rx,Ry,a,b,thickness,nu,E):
    '''
    Returns the functions to calculate the stress and strain tensor field 
    components as a function of position on the crystal wafer for an isotropic
    rectangular toroidally bent crystal analyser. The edges of the rectangular
    wafer are assumed to be parallel to the axes of curvature.

    Parameters
    ----------
    Rx : float
        Meridional bending radius
    Ry : float
        Sagittal bending radius
    a : float
        Meridional wafer dimension
    b : float
        Sagittal wafer dimension
    thickness : float
        Thickness of the wafer
    nu : float
        Poisson's ratio
    E : float
        Young's modulus. Units determine the units of the returned stress 
        tensor components.

    For sensible output, the physical units of Rx, Ry, a, b, and thickness have 
    to be the same.

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

    #Geometry factor used in the stress components
    g = 8 + 10*(a**2/b**2 + b**2/a**2) + (1-nu)*(a**2/b**2 - b**2/a**2)**2

    #Define stress functions
    stress = {}
            
    def sigma_xx(x,y):
        stress = E/(g*Rx*Ry)*(a**2/12 - x**2 + (b**2/12 - y**2)
                              *((1+nu)/2 + 5*a**2/b**2 + (1-nu)/2*a**4/b**4))
        stress[np.logical_or(x**2 > a**2/4, y**2 > b**2/4)] = np.nan
        return stress

    def sigma_yy(x,y):
        stress = E/(g*Rx*Ry)*(b**2/12 - y**2 + (a**2/12 - x**2)
                              *((1+nu)/2 + 5*b**2/a**2 + (1-nu)/2*b**4/a**4))
        stress[np.logical_or(x**2 > a**2/4, y**2 > b**2/4)] = np.nan
        return stress

    def sigma_xy(x,y):
        stress = 2*E/(g*Rx*Ry)*x*y
        stress[np.logical_or(x**2 > a**2/4, y**2 > b**2/4)] = np.nan
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
        return (sigma_xx - nu*sigma_yy)/E

    def epsilon_yy(x,y):
        return (sigma_yy - nu*sigma_xx)/E

    def epsilon_xy(x,y):
        return (1+nu)*sigma_xy/E

    def epsilon_xz(x,y):
        strain = np.zeros(np.array(x).shape)
        strain[np.logical_or(x**2 > a**2/4, y**2 > b**2/4)] = np.nan
        return strain

    def epsilon_yz(x,y):
        strain = np.zeros(np.array(x).shape)
        strain[np.logical_or(x**2 > a**2/4, y**2 > b**2/4)] = np.nan
        return strain

    def epsilon_zz(x,y):
        return -nu*(sigma_xx + sigma_yy)/E

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
        P = -E*thickness/(g*Rx**2*Ry**2)*(
             (a**2/12 - x**2)*(Rx*((1+nu)/2 + 5*b**2/a**2 + (1-nu)/2*b**4/a**4) + Ry)
            +(b**2/12 - y**2)*(Ry*((1+nu)/2 + 5*a**2/b**2 + (1-nu)/2*a**4/b**4) + Rx))
        P[np.logical_or(x**2 > a**2/4, y**2 > b**2/4)] = np.nan
        return P
    
    return stress, strain, contact_force
