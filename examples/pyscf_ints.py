import pyscf


def molecule(verbose=0):

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open('LiH.xyz') as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = 'sto-3g'
    mol.charge = 0
    mol.spin = 0

    return mol


if __name__ == '__main__':

    mol = molecule()
    mol.build()
    derovl = mol.intor('cint1e_ipovlp_sph', comp=3)
    print(derovl)

