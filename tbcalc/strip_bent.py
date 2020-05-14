# -*- coding: utf-8 -*-
"""
This file defines the function to construct the deformation field of a 
strip-bent analyser.

Created on Thu May 14 14:43:10 2020

@author: aripekka
"""
from __future__ import division, print_function
from .transverse_deformation import isotropic_rectangular, anisotropic_rectangular
import numpy as np

                               
def strip_coordinates(diameter, strip_width, strip_orientation, center_strip, lateral_strips):
    '''
    Calculates the coordinates and dimensions of strips on a circular analyser
    surface.

    Parameters
    ----------
    diameter : float
        Diameter of the analyser. Given in the same units as strip_width. 

    strip_width : float
        Width of the strips. Given in the same units as diameter.

    strip_orientation : str
        Orientation of the strips. Either 'meridional' or 'sagittal'.

    center_strip : bool
        If True, the analyser has a strip going through the middle of the 
        crystal (the center of this strip coincides with the center of the 
        analyser). If False, then the center of the crystal is assumed to fall
        between two strips.
    
    lateral_strips : 'wide' or 'narrow'
        The strip width is not necessarily a multiple of the analyser diameter
        which means that the most lateral strips have to be either wider ('wide')
        or narrower ('narrow') than the rest of the strips.

    Returns
    -------
    
    strips : list
        List of tuples of form (x,y,a,b) where x and y are the coordinates of 
        strip centers, and a and b are the strip side lengths. All are floats
        in the same physical units as the inputs diameter and strip_width. Note
        that the list is not ordered.

    '''

    #For calculating the positions, let us define new coordinates (u,v) of 
    #which u is in the direction of long edges of the strips and v is 
    #perpendicular to that. Thus if orientation is meridional then (u,v) = (x,y)
    #and if sagittal then (u,v) = (y,-x)

    strips_uv = [] #tuples (u,v,width,height)

    if center_strip:
        strips_uv.append((0,0,diameter,strip_width))

        next_center_v = strip_width
    else:
        next_center_v = 0.5*strip_width

    #next_center_v is the position of the center of the next strip to be added.
    #Since the analyser is symmetric with respect to the center we also add
    #another strip at -next_center_v

    #Calculate the number of strips that fit the remaining (half)space     
    n_strips = (0.5*diameter-next_center_v+0.5*strip_width)/strip_width

    #Calculate the number of strips excluding the lateral pieces
    if lateral_strips == 'narrow':
        n_normal_strips = int(n_strips)
    else:
        n_normal_strips = int(n_strips) - 1   

    #Calculate the positions of the normal strips
    for i in range(n_normal_strips):

        #Calculate the strip length so that it fully covers its section of the 
        #circular area
        strip_length = 2*np.sqrt(diameter**2/4 - (next_center_v-0.5*strip_width)**2)
        
        strips_uv.append((0,next_center_v,strip_length,strip_width))
        strips_uv.append((0,-next_center_v,strip_length,strip_width))
        
        next_center_v += strip_width
    
    #Handle the lateral pieces separately
    
    lateral_bottom_v = next_center_v - 0.5*strip_width

    strip_width = diameter/2 - lateral_bottom_v
    strip_length = 2*np.sqrt(diameter**2/4 - lateral_bottom_v**2)

    next_center_v = lateral_bottom_v + 0.5*strip_width

    strips_uv.append((0,next_center_v,strip_length,strip_width))
    strips_uv.append((0,-next_center_v,strip_length,strip_width))

    #Set orientation
    if strip_orientation == 'meridional':
        strips = strips_uv
    else:
        strips = []
        for i in strips_uv:
            strips.append((i[1],-i[0],i[3],i[2]))

    return strips            

def combine_strips(strips, stresses, strains, c_forces, diameter):
    '''
    Combines the stress, strain, and contact force functions of individual
    strips.
    '''
    
    stress = {}

    def stress_ij(X,Y,ij):
        stress_sum = np.empty(X.shape)
        stress_sum[:,:] = np.nan

        for i in range(len(strips)):
            temp = stresses[i][ij](X-strips[i][0],Y-strips[i][1])
            stress_sum[np.logical_not(np.isnan(temp))] = temp[np.logical_not(np.isnan(temp))]

        stress_sum[X**2 + Y**2 > diameter**2/4] = np.nan
            
        return stress_sum


    stress['xx'] = lambda X,Y: stress_ij(X,Y,'xx')
    stress['xy'] = lambda X,Y: stress_ij(X,Y,'xy')
    stress['yy'] = lambda X,Y: stress_ij(X,Y,'yy')

    #Add symmetric indices
    stress['yx'] = stress['xy']    

    #Add numerical indices
    stress[11] = stress['xx']
    stress[12] = stress['xy']
    stress[21] = stress['yx']    
    stress[22] = stress['yy']

    strain = {}

    def strain_ij(X,Y,ij):
       
        strain_sum = np.empty(X.shape)
        strain_sum[:,:] = np.nan

        for i in range(len(strips)):
            temp = strains[i][ij](X-strips[i][0],Y-strips[i][1])
            strain_sum[np.logical_not(np.isnan(temp))] = temp[np.logical_not(np.isnan(temp))]
 
        strain_sum[X**2 + Y**2 > diameter**2/4] = np.nan           
 
        return strain_sum
   
    strain['xx'] = lambda X,Y: strain_ij(X,Y,'xx')
    strain['xy'] = lambda X,Y: strain_ij(X,Y,'xy')
    strain['yy'] = lambda X,Y: strain_ij(X,Y,'yy')
    strain['xz'] = lambda X,Y: strain_ij(X,Y,'xz')
    strain['yz'] = lambda X,Y: strain_ij(X,Y,'yz')
    strain['zz'] = lambda X,Y: strain_ij(X,Y,'zz')

    #Add symmetric indices
    strain['yx'] = strain['xy']    
    strain['zx'] = strain['xz']    
    strain['zy'] = strain['yz']    
    
    #Add numerical indices
    strain[11] = strain['xx']
    strain[12] = strain['xy']
    strain[21] = strain['yx']    
    strain[22] = strain['yy']
    strain[13] = strain['xz']
    strain[31] = strain['zx']
    strain[23] = strain['yz']    
    strain[32] = strain['zy']    
    strain[33] = strain['zz']


    def contact_force(X,Y):
       
        cforce_sum = np.empty(X.shape)
        cforce_sum[:,:] = np.nan

        for i in range(len(strips)):
            temp = c_forces[i](X-strips[i][0],Y-strips[i][1])
            cforce_sum[np.logical_not(np.isnan(temp))] = temp[np.logical_not(np.isnan(temp))]

        cforce_sum[X**2 + Y**2 > diameter**2/4] = np.nan       
            
        return cforce_sum

    return stress, strain, contact_force


def isotropic_strip_bent_deformation(Rx, Ry, diameter, strip_width, strip_orientation, 
                                     center_strip, lateral_strips, thickness, nu, E):
    
    #Get the center positions and dimensions of the strips
    strips = strip_coordinates(diameter, strip_width, strip_orientation, center_strip, lateral_strips)

    #Get the functions calculating stress, strain and contact forces for individual
    #strips
    stresses = []
    strains = []
    c_forces = []
    
    for i in range(len(strips)):        
        x, y, a, b = strips[i]
        stress, strain, c_force = isotropic_rectangular(Rx,Ry,a,b,thickness,nu,E)
        stresses.append(stress)
        strains.append(strain)
        c_forces.append(c_force)

    #Combine the single strip functions
    return combine_strips(strips, stresses, strains, c_forces, diameter)

def anisotropic_strip_bent_deformation(Rx, Ry, diameter, strip_width, strip_orientation, 
                                       center_strip, lateral_strips, thickness, S):
    
    #Get the center positions and dimensions of the strips
    strips = strip_coordinates(diameter, strip_width, strip_orientation, center_strip, lateral_strips)

    #Get the functions calculating stress, strain and contact forces for individual
    #strips
    stresses = []
    strains = []
    c_forces = []
    
    for i in range(len(strips)):        
        x, y, a, b = strips[i]
        stress, strain, c_force = anisotropic_rectangular(Rx,Ry,a,b,thickness, S)
        stresses.append(stress)
        strains.append(strain)
        c_forces.append(c_force)

    #Combine the single strip functions
    return combine_strips(strips, stresses, strains, c_forces, diameter)