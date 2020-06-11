# tbcalc
Package to calculate X-ray diffraction curves toroidally bent crystal analysers (TBCAs). For derivative work, please cite

Ari-Pekka Honkanen and Simo Huotari, "General method to calculate the elastic deformation and X-ray diffraction properties of bent crystal wafers" (2020) Submitted to IUCrJ. https://arxiv.org/abs/2006.04952

## Installation 
tbcalc can be installed with pip with the following command:

```
pip install tbcalc
```

However, note that tbcalc is built on top of 1D Takagi-Taupin solver package PyTTE https://github.com/aripekka/pyTTE which relies on xraylib 4.0.0 which can not be installed with pip. See https://github.com/tschoonj/xraylib for instructions.

## Example of use

The following example calculates the reflectivity curve for a circular Si(660) analyser with the diameter of 100 mm and thickness 150 um. The wafer is spherically bent with the bending radius of 1 m. 

An energy scan is performed with automatically determined limits at 80 degrees with a sigma polarized beam. The incident bandwidth is assumed to be gaussian with the standard deviation of 300 meV. The analyser is assumed to be masked with a mask with a 60 mm diameter circular aperture.

```
from tbcalc import Analyser, TTscan, Quantity
import numpy as np

ana = Analyser(crystal = 'Si', hkl = [6,6,0], thickness = Quantity(150, 'um'), R = Quantity(1, 'm'), diameter = Quantity(100, 'mm'))

scan = TTscan(constant = Quantity(80,'deg'), scan = 150, polarization = 'sigma')

#Coordinates for mask are always in mm 
def mask_60mm(X,Y):
  return X**2 + Y**2 < 30**2

e_bw = np.linspace(-1000,1000,200)
I_bw = np.exp(-e_bw**2/(2*300))

ana.run(scan, mask = mask_60mm, incident_bandwidth = (Quantity(e_bw,'meV'), I_bw))

#Get the calculated curve
E, I = ana.solution['scan'].in_units('eV'), ana.solution['total_curve']

```

For more instructions and examples see the docstrings of the classes and the Jupyter notebooks in examples/.
