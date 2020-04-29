# -*- coding: utf-8 -*-
from pyTTE import TakagiTaupin, Quantity


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

    S : 6x6 array wrapped in a Quantity instance of type pressure^-1
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
    
    
    '''    


    def __init__(self,filepath=None, **kwargs):
        pass