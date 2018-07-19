import numpy as np

from pyints.numerical import fermi_contact

za = 1.8
zb = 2.0
ra = np.asarray([0.0, 0.0, 0.0])
rb = np.asarray([0.5, 0.8, -0.2])
rc = np.asarray([0.5, 0.8, 0.2])

la = np.asarray([0, 0, 0])
lb = np.asarray([0, 0, 0])

prefac = 4.0 * np.pi / 3.0

res = fermi_contact(za, la, ra, zb, lb, rb, rc)
print(prefac * res)
