"""Generic classes that define the properties of each integral type
without knowing an interface to each program."""


class Integral:
    pass


class Integral1e2c(Integral):
    pass


class Integral2e4c(Integral):
    pass


class Overlap(Integral1e2c):
    pass


class Kinetic(Integral1e2c):
    pass


class Potential(Integral1e2c):
    pass


class FermiContact(Integral1e2c):
    pass


FC = FermiContact


class SpinDipole(Integral1e2c):
    pass


SD = SpinDipole


class DipoleLength(Integral1e2c):
    pass


Dipole = DipoleLength


class SecondMoment(Integral1e2c):
    pass


class Quadrupole(Integral1e2c):
    pass


class TracelessQuadrupole(Integral1e2c):
    pass


class MultipoleLength(Integral1e2c):
    pass


Multipole = MultipoleLength


class DipoleVelocity(Integral1e2c):
    pass


Nabla = DipoleVelocity


class AngularMomentum(Integral1e2c):
    pass


class OverlapDer1(Integral1e2c):
    pass


class KineticDer1(Integral1e2c):
    pass


class PotentialDer1(Integral1e2c):
    pass


class Repulsion(Integral2e4c):
    pass


class SpinOrbitExact(Integral2e4c):
    pass
