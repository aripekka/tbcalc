# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np

def johann_error(X,Y,Rx,Ry,th,energy=None):
    '''
    Calculates the Johann error for given x-coordinates (dispersive direction)
    in angle or energy domain, depending whether energy is given.

    Parameters
    ----------
    
    X, Y : Numpy array
        Coordinates along the meridional and sagittal directions (X,Y=0 coincides 
        with the center of the crystal)

    Rx, Ry : float
        Meridional and sagittal bending radii in the same units as X and Y

    th : float 
        Incidence angle of X-ray with respect to the crystal surface at the
        center of the crystal in radians

    energy : float or None 
        Energy of photons in units of preference.
        
    Returns
    -------
    
    Change in the diffraction condition in terms of energy (if input energy is
    float) or angle (if input energy is None).

    '''

    if energy is None:
        return  X**2/(2*Rx**2*np.tan(th)) - (Rx - Ry)*(Rx*np.sin(th)**2 - Ry)/(2*Rx*Ry*np.sin(th)*np.cos(th))*Y**2
    else:
        return -X**2/(2*Rx**2*np.tan(th)**2)*energy + (Rx - Ry)*(Rx*np.sin(th)**2 - Ry)/(2*Rx*Ry*np.sin(th)**2)*Y**2*energy