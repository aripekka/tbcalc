# -*- coding: utf-8 -*-
"""
Created on Sat May  9 00:44:37 2020

@author: aripekka
"""

import numpy as np

def cartesian_tensors_to_cylindrical(tensor):
    '''
    Converts the Cartesian components of the transverse stress and strain tensors 
    to cylindrical components. See the technical documentation for the detailed
    explanation and formulas.

    Parameters
    ----------
    tensor : dict
        Either stress or strain tensor dictionaries as returned by functions
        defined in tbcalc.transverse_deformation.

    Returns
    -------
    tensor_cyl : dict
        Functions of r and phi returning the components of the tensor in cylindrical 
        coordinates. The dictionary keys are of form i+j where i,j = 'r', 'phi' 
        or 'z' ('z' not available for the stress tensor). The length unit for
        r is as for x and y and phi is given in radians.

    '''
    
    tensor_cyl = {}
    
    tensor_cyl['rr'] = lambda r,phi : ( tensor['xx'](r*np.cos(phi),r*np.sin(phi))*np.cos(phi)**2
                                      + 2*tensor['xy'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi)*np.cos(phi)
                                      + tensor['yy'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi)**2 )

    tensor_cyl['rphi'] = lambda r,phi : (- tensor['xx'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi)*np.cos(phi)
                                         + tensor['xy'](r*np.cos(phi),r*np.sin(phi))*(np.cos(phi)**2 - np.sin(phi)**2)
                                         + tensor['yy'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi)*np.cos(phi) )

    tensor_cyl['phiphi'] = lambda r,phi : ( tensor['xx'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi)**2
                                          - 2*tensor['xy'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi)*np.cos(phi)
                                          + tensor['yy'](r*np.cos(phi),r*np.sin(phi))*np.cos(phi)**2 )

    tensor_cyl['phir'] = tensor_cyl['rphi']

    if 'zz' in tensor:
        tensor_cyl['rz'] = lambda r, phi : ( tensor['xz'](r*np.cos(phi),r*np.sin(phi))*np.cos(phi)
                                           + tensor['yz'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi))

        tensor_cyl['phiz'] = lambda r, phi : (- tensor['xz'](r*np.cos(phi),r*np.sin(phi))*np.sin(phi)
                                              + tensor['yz'](r*np.cos(phi),r*np.sin(phi))*np.cos(phi))

        tensor_cyl['zz'] = lambda r, phi : tensor['zz'](r*np.cos(phi),r*np.sin(phi))

        tensor_cyl['zr'] = tensor_cyl['rz']
        tensor_cyl['zphi'] = tensor_cyl['phiz']
        
    return tensor_cyl