# -*- coding: utf-8 -*-
"""
Tests for the Analyser class. Run with pytest.

Created on Wed Apr 29 17:28:01 2020

@author: aripekka
"""

import sys
import os.path

sys.path.insert(1, os.path.join(os.path.dirname(__file__),'..'))

from pyTTE import Quantity
from tbcalc import Analyser 

def test_init():

    #Correct inputs
    correct_input = []
    
    #circular analyser
    correct_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         }, 'circular'])

    #Spherical bending
    correct_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'R' : Quantity(1,'m'),
                         'diameter' : Quantity(100,'mm'),
                         }, 'circular'])

    correct_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'R' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         }, 'circular'])


    #rectangular analyser
    correct_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'a' : Quantity(100,'mm'),
                         'b' : Quantity(50,'mm'),
                         }, 'rectangular'])

    #strip-bent analyser
    correct_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         'strip_width' : Quantity(15,'mm'),
                         },'strip-bent'])    

    for ci in correct_input:
        ana = Analyser(**ci[0])

        assert ana.geometry_info['wafer_shape'] == ci[1]

        #test the initialization of the geometry_info dict
        if ci[1] == 'circular':
            assert set(ana.geometry_info) == {'wafer_shape','diameter'}
            assert ana.geometry_info['diameter'].value == ci[0]['diameter'].value                      
        elif ci[1] == 'rectangular':
            assert set(ana.geometry_info) == {'wafer_shape','a','b'}        
            assert ana.geometry_info['a'].value == ci[0]['a'].value            
            assert ana.geometry_info['b'].value == ci[0]['b'].value            

        elif ci[1] == 'strip-bent':
            assert set(ana.geometry_info) == {'wafer_shape','diameter','strip_width','strip_orientation'}        
            assert ana.geometry_info['diameter'].value == ci[0]['diameter'].value                
            assert ana.geometry_info['strip_width'].value == ci[0]['strip_width'].value    
            assert ana.geometry_info['strip_orientation'] == ci[0].get('strip_orientation','meridional') 

    wrong_input = []


    #Missing bending radii
    wrong_input.append({
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'diameter' : Quantity(100,'mm'),
                         })

    #Missing bending radii
    wrong_input.append({
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Ry' : Quantity(0.5,'m'),                         
                         'diameter' : Quantity(100,'mm'),
                         })       

    #Missing bending radii
    wrong_input.append({
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(0.5,'m'),                         
                         'diameter' : Quantity(100,'mm'),
                         })      

    for wi in wrong_input:
        try:
            ana = Analyser(**wi)
            assert False
        except:
            pass
    