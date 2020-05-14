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

    #strip-bent analyser
    correct_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         'strip_width' : Quantity(15,'mm'),
                         'center_strip': False,
                         },'strip-bent'])    

    #strip-bent analyser
    correct_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         'strip_width' : Quantity(15,'mm'),
                         'lateral_strips' : 'narrow',
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
            assert set(ana.geometry_info) == {'wafer_shape','diameter','strip_width',
                                              'strip_orientation','center_strip', 'lateral_strips'}        
            assert ana.geometry_info['diameter'].value == ci[0]['diameter'].value                
            assert ana.geometry_info['strip_width'].value == ci[0]['strip_width'].value    
            assert ana.geometry_info['strip_orientation'] == ci[0].get('strip_orientation','meridional')
            assert ana.geometry_info['center_strip'] == ci[0].get('center_strip',True)
            assert ana.geometry_info['lateral_strips'] == ci[0].get('lateral_strips','wide')

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

    #Missing crystal paramter
    wrong_input.append({
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'R' : Quantity(0.5,'m'),                         
                         'diameter' : Quantity(100,'mm'),
                         })      

    #strip-bent analyser
    wrong_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         'strip_width' : Quantity(15,'mm'),
                         'lateral_strips' : 'sdsd',
                         },'strip-bent'])   

    #strip-bent analyser
    wrong_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         'strip_width' : Quantity(15,'mm'),
                         'lateral_strips' : False,
                         },'strip-bent'])   

    #strip-bent analyser
    wrong_input.append([{
                         'crystal' : 'Si',
                         'hkl' : [4,0,0],
                         'thickness' : Quantity(150,'um'),
                         'Rx' : Quantity(1,'m'),
                         'Ry' : Quantity(0.5,'m'),
                         'diameter' : Quantity(100,'mm'),
                         'strip_width' : Quantity(15,'mm'),
                         'center_strip' : 'scs',
                         },'strip-bent'])   

    wrong_input.append([{
                     'crystal' : 'Si',
                     'hkl' : [4,0,0],
                     'thidckness' : Quantity(150,'um'),
                     'Rx' : Quantity(1,'m'),
                     'Ry' : Quantity(0.5,'m'),
                     'diameter' : Quantity(100,'mm'),
                     'strip_width' : Quantity(15,'mm'),
                     },'strip-bent'])   

    for wi in wrong_input:
        try:
            ana = Analyser(**wi)
            assert False
        except:
            pass

def test_init_TTcrystal():
    '''
    Check that the TTcrystal object in Analyser is initialized correctly
    '''

    inp = []
    inp.append({
                'crystal' : 'Ge',
                'hkl' : [5,3,1],
                'thickness' : Quantity(150,'um'),
                'Rx' : Quantity(1,'m'),
                'Ry' : Quantity(0.5,'m'),
                'diameter'  : Quantity(100,'mm'),
                'asymmetry' : Quantity(5,'deg'),
                'in_plane_rotation' : Quantity(45,'deg'),
                'debye_waller'  : 0.8,
                })


    for i in inp:
        ana = Analyser(**i)
    
        assert ana.crystal_object.crystal_data['name'] == i['crystal']
        assert ana.crystal_object.hkl == i['hkl']
        assert ana.crystal_object.debye_waller == i['debye_waller']

        assert ana.crystal_object.thickness.value == i['thickness'].value
        assert ana.crystal_object.asymmetry.value == i['asymmetry'].value
        assert ana.crystal_object.in_plane_rotation.value == i['in_plane_rotation'].value
        assert ana.crystal_object.Rx.value == i['Rx'].value
        assert ana.crystal_object.Ry.value == i['Ry'].value
        
        assert ana.crystal_object.fix_to_axes == 'shape'

def test_deformation():
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

        try:
            ana.calculate_deformation()
        except NotImplementedError:
            pass