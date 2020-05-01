# -*- coding: utf-8 -*-
from pyTTE import TakagiTaupin, Quantity
import numpy as np

class Analyser:
    '''
    Analyser is the main class of the tbcalc package that contains all the 
    information and functions needed compute diffraction curves of toroidally
    bent crystal analysers.
    
    Currently circular wafers, rectangular wafers, and strip-bent analysers are
    supported. The type of the analyser is determined at the initialization 
    based on the input parameters given:
        
        Circular analyser    : diameter
        Rectangular analyser : a, b
        Strip-bent analyser  : diameter, strip_width, strip_orientation

    Any other combination of the parameters above (e.g. diameter and a) will 
    raise an error.

    Parameters
    ----------

    filepath : str
        Path to the file with the analyser parameters
    
    *OR*

    crystal : str (all analysers)
        String representation of the crystal in compliance with xraylib

    hkl : list, tuple, or 1D array of size 3 (all analysers)
        Miller indices of the reflection (ints or floats)

    thickness : pyTTE.Quantity of type length (all analysers)
        Thickness of the crystal wafer e.g. Quantity(300,'um')

    Rx, Ry : pyTTE.Quantity of type length (all analysers if not R given)
        Meridional and sagittal bending radii for toroidal bending wrapped in 
        Quantity instances e.g. Quantity(1,'m'). If omitted, defaults to inf 
        (no bending). Overridden by R.         
       
    R : pyTTE.Quantity of type length (all analysers, optional)
        Bending radius for spherical bending wrapped in Quantity instance. 
        Overrides Rx and Ry.

    diameter : pyTTE.Quantity of type length (circular and strip-bent analysers)
        Diameter of the analyser e.g. Quantity(100,'mm').

    a, b : pyTTE.Quantity of type length (rectangular analysers)
        Meridional and sagittal dimensions of the rectangular wafer.

    strip_width : pyTTE.Quantity of type length (strip_bent analysers)
        Width of the strips of a strip-bent analyser.

    strip_orientation : str (strip_bent analysers, optional)
        Orientation of the strips. The direction of the cuts can be either
        'meridional' or 'sagittal'. If not given, defaults to 'meridional'.

    asymmetry : pyTTE.Quantity of type angle (all analysers, optional)
        Clockwise-positive asymmetry angle wrapped in a Quantity instance.
        0 deg for symmetric Bragg case (default), 90 deg for symmetric Laue

    in_plane_rotation : pyTTE.Quantity of type angle OR a list of size 3
                        (all analysers, optional)
        Counterclockwise-positive rotation of the crystal directions about the
        normal vector of (hkl) wrapped in a Quantity instance of type angle
        OR a crystal direction [q,r,s] corresponding to a direct space vector
        R = q*a1 + r*a2 + s*a3 that together with the crystal will be rotated 
        about the hkl vector so that its component perpendicular to the normal 
        of (hkl) will be aligned with the y-axis. Will raise an error if 
        R || hkl.
        
    debye_waller : float in range [0, 1] (all analysers, optional)
        The Debye-Waller factor to account for the thermal motion. Definined as
        exp(-0.5 * h^2 * <u^2>), where h is the reciprocal lattice vector 
        corresponding to (hkl) and <u^2> is the expectation value of mean 
        displacement of atoms parallel to h. Currently assumes that all atoms
        share the same <u^2>. Defaults to 1 (= 0 K).

    S : 6x6 array wrapped in a pyTTE.Quantity instance of type pressure^-1
        The compliance matrix in the Voigt notation. Overrides the default 
        compliance matrix given by elastic_tensors and any user inputs for E 
        and nu. (all analysers, optional)
                       
        Note that S is supposed to be in the Cartesian coordinate system aligned
        with the conventional unit vectors before any rotations i.e. x || a_1 
        and a_2 is in the xy-plane. For rectagular systems this means that the 
        Cartesian basis is aligned with the unit vectors. 

        If an input file is used, the non-zero elements of the compliance matrix
        in the upper triangle and on the diagonal should be given in the units 
        GPa^-1 (order doesn't matter). Any lower triangle inputs will be omitted 
        as they are obtained by symmetry from the upper triangle. 

        Example input: 
            S11  0.00723
            S22  0.00723
            S33  0.00723
            S12 -0.00214
            etc.

    E : pyTTE.Quantity of type pressure (all analysers, optional)
        Young's modulus for isotropic material. Overrides the default compliance 
        matrix. Neglected if S is given. Required with nu. Can have an arbitrary 
        value for diffraction calculations but has to be correct for stress 
        fields.
        
    nu : float (all analysers, optional)
        Poisson's ratio for isotropic material. Overrides the default compliance 
        matrix. Neglected if S is given. Requires that E also given.


    Attributes
    ----------

    crystal_object: TTcrystal
        A TTcrystal object constructed from the input parameters
        
    geometry_info: dict
        Contains the information of the wafer geometry and analyser type. Has 
        the following keywords:

            'wafer_shape' (all analysers) : str
                Either 'circular', 'rectangular', 'strip-bent' or 'custom' 
                (the last one is reserved for future used)

            'diameter' : pyTTE.Quantity of type length (circular and strip-bent)
                The diameter of the analyser.
                
            'a', 'b' : pyTTE.Quantity of type length (rectangular)
                Meridional and sagittal dimensions of the rectangular wafer.

            'strip_width' : pyTTE.Quantity of type length (strip-bent)
                Width of the strips of the strip-bent analyser
                
            'strip_orientation' : str (strip-bent)
                Orientation of the long dimension of the strips with respect to
                to diffraction plane. Either 'meridional' or 'sagittal'
    
    '''    


    def __init__(self,filepath=None, **kwargs):
            
        params = {}

        if filepath is not None:
            
            #####################################
            #Read crystal parameters from a file#
            #####################################

            #Overwrite possible kwargs 
            kwargs = {}

            with open(filepath,'r') as f:
                lines = f.readlines()

            #Boolean to check if elements of the compliance matrix are given
            is_S_given = False
            S_matrix = np.zeros((6,6))            

            #check and parse parameters
            for line in lines:
                line = line.strip()
                if len(line) > 0 and not line[0] == '#':  #skip empty and comment lines
                    ls = line.split() 
                    if ls[0] == 'crystal' and len(ls) == 2:
                        kwargs['crystal'] = ls[1]
                    elif ls[0] == 'hkl' and len(ls) == 4:
                        kwargs['hkl'] = [int(ls[1]),int(ls[2]),int(ls[3])]
                    elif ls[0] in ['Rx', 'Ry']:
                        if len(ls) == 3:
                            kwargs[ls[0]] = Quantity(float(ls[1]),ls[2])
                        elif len(ls) == 2:
                            kwargs[ls[0]] = ls[1]
                        else:
                            print('Skipped an invalid line in the file: ' + line)                           
                    elif ls[0] == 'R' and len(ls) == 3:                        
                        kwargs['Rx'] = Quantity(float(ls[1]),ls[2])
                        kwargs['Ry'] = Quantity(float(ls[1]),ls[2])         
                    elif ls[0] == 'in_plane_rotation':
                        if len(ls) == 4:
                            kwargs['in_plane_rotation'] = [float(ls[1]),float(ls[2]),float(ls[3])]
                        elif  len(ls) == 3:
                            kwargs['in_plane_rotation'] = Quantity(float(ls[1]),ls[2])
                        else:
                            print('Skipped an invalid line in the file: ' + line)
                    elif ls[0] == 'debye_waller' and len(ls) == 2:
                        kwargs['debye_waller'] = float(ls[1])
                    elif ls[0] in ['thickness','diameter','a','b','strip_width', 'asymmetry', 'E'] and len(ls) == 3:
                        kwargs[ls[0]] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'nu' and len(ls) == 2:
                        kwargs['nu'] = float(ls[1])
                    elif ls[0][0] == 'S' and len(ls[0]) == 3 and len(ls) == 2:
                        is_S_given = True
                        i = int(ls[0][1])-1
                        j = int(ls[0][2])-1
                        if i > j:
                            print('Omitted the lower triangle element ' + ls[0] + '.')
                        else:
                            S_matrix[i,j] = float(ls[1])
                            S_matrix[j,i] = float(ls[1])           
                    else:
                        print('Skipped an invalid line in the file: ' + line)

            if is_S_given:
                #Finalize the S matrix
                kwargs['S'] = Quantity(S_matrix,'GPa^-1') 

        ###################################################
        #Check the presence of the required crystal inputs#
        ###################################################
        
        try:
            for k in ['crystal','hkl','thickness']:
                params[k] = kwargs[k]
        except:
            raise KeyError('At least one of the required keywords crystal,'\
                          +'hkl, thickness, Rx, Ry, or R is missing!')

        if 'Rx' in kwargs and 'Ry' in kwargs:
            params['Rx'] = kwargs['Rx']
            params['Ry'] = kwargs['Ry']        
        elif 'R' in kwargs:  
            if 'Rx' in kwargs or 'Rx' in kwargs:
                print('Warning! Rx and/or Ry given but overridden by R.')
            params['Rx'] = kwargs['R']
            params['Ry'] = kwargs['R']
        else:
            raise KeyError('The bending radii Rx or Ry, or R are missing!')


        #Optional keywords       
        for k in ['asymmetry','in_plane_rotation']:
            params[k] = kwargs.get(k, Quantity(0,'deg'))

        params['debye_waller'] = kwargs.get('debye_waller', 1.0)

        for k in ['S','E','nu']:
            params[k] = kwargs.get(k, None)

        #Check that if either E or nu is given, then the other one is also
        if (params['E'] is not None) ^ (params['nu'] is not None):
            raise KeyError('Both E and nu required for isotropic material!')


        #Check and set the wafer geometry
        self.geometry_info = {}
   
        if 'diameter' in kwargs:
            #Handle circular and strip-bent geometries     
            
            #Check the validity of given diameter
            diameter = kwargs['diameter']
            if isinstance(diameter,Quantity) and diameter.type() == 'length' and diameter.value.size == 1:            
                self.geometry_info['diameter'] = diameter
            else:
                raise TypeError('diameter has to be a single value in pyTTE.Quantity of type length!')
                
            #check the presence of additional keywords
            if 'a' in kwargs or 'b' in kwargs:
                raise KeyError('Keywords a or b can not be given with the keyword diameter!')
            elif 'strip_width' in kwargs:
                
                strip_width = kwargs['strip_width']
                if isinstance(strip_width,Quantity) and strip_width.type() == 'length' and strip_width.value.size == 1:            
                    self.geometry_info['strip_width'] = strip_width
                else:
                    raise TypeError('strip_width has to be a single value in pyTTE.Quantity of type length!')
               
                self.geometry_info['strip_orientation'] = kwargs.get('strip_orientation', 'meridional').lower()

                if self.geometry_info['strip_orientation'] not in ['meridional','sagittal']:
                    raise TypeError("strip_orientation has to be either 'meridional' or 'sagittal'!" )

                self.geometry_info['wafer_shape'] = 'strip-bent'

            else:
                self.geometry_info['wafer_shape'] = 'circular'
        else:
            #Handle rectangular geometry
            if 'a' in kwargs and 'b' in kwargs:
                if 'strip_width' in kwargs or 'strip_orientation' in kwargs:
                    raise KeyError('Keywords strip_width and/or strip_orientation can not be given together with a and b!')

                if isinstance(kwargs['a'],Quantity) and kwargs['a'].type() == 'length' and kwargs['a'].value.size == 1:            
                    self.geometry_info['a'] = kwargs['a']
                    if isinstance(kwargs['b'],Quantity) and kwargs['b'].type() == 'length' and kwargs['b'].value.size == 1:            
                        self.geometry_info['b'] = kwargs['b']
                    else:
                        raise TypeError('b has to be a single value in pyTTE.Quantity of type length!')                                       
                else:
                    raise TypeError('a has to be a single value in pyTTE.Quantity of type length!')
                
                self.geometry_info['wafer_shape'] = 'rectangular'
            else:
                raise KeyError('Both keywords a and b are required!')                
            