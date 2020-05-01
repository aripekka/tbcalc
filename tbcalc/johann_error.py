# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np

def johann_error(X,Rx,th,energy=None):
    '''
    Calculates the Johann error for given x-coordinates (dispersive direction)
    in angle or energy domain, depending whether energy is given.

    Parameters
    ----------
    
    X : Numpy array
        Coordinates along the dispersion direction (X=0 coincides with the
        center of the crystal)

    Rx : float
        Meridional bending radius in the same units as X

    th : Incidence angle of X-ray with respect to the crystal surface at the
         center of the crystal

    energy : float or None 
        Energy of photons in units of preference.
        
    Returns
    -------
    
    Change in the diffraction condition in terms of energy (if input energy is
    float) or angle (if input energy is None).

    '''
    if np.tan(th) == 0:
        return np.zeros(X.shape)
    elif energy is None:
        return  X**2/(2*Rx**2*np.tan(th))
    else:
        return -X**2/(2*Rx**2*np.tan(th)**2)*energy