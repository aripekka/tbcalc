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
        
        assert ana.type == ci[1]