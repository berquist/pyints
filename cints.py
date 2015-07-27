import ctypes
import pyscf.lib

_cint = pyscf.lib.load_library('libcvhf')
_cint.CINTcgto_cart.restype = ctypes.c_int
_cint.CINTcgto_spheric.restype = ctypes.c_int
_cint.CINTcgto_spinor.restype = ctypes.c_int

from pyints.gto.moleintor import getints


### overlap (S) integrals

def makeS(mol):
    ints = getints('cint1e_ovlp_cart', mol._atm, mol._bas, mol._env)
    return ints

### kinetic energy (T) integrals

def makeT(mol):
    ints = getints('cint1e_kin_cart', mol._atm, mol._bas, mol._env)
    return ints

### nuclear attraction (V) integrals

def makeV(mol):
    ints = getints('cint1e_nuc_cart', mol._atm, mol._bas, mol._env)
    return ints
