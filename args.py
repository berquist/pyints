def getargs():
    """Get command-line arguments."""

    import argparse

    parser = argparse.ArgumentParser()

    # One-electron integrals
    parser.add_argument('--S',
                        action='store_true',
                        help="""Calculate overlap integrals.""")
    parser.add_argument('--T',
                        action='store_true',
                        help="""Calculate kinetic energy integrals.""")
    parser.add_argument('--V',
                        action='store_true',
                        help="""Calculate nuclear attraction integrals.""")
    parser.add_argument('--M',
                        action='store_true',
                        help="""Calculate Cartesian moment integrals.""")
    parser.add_argument('--order-M',
                        nargs=3,
                        type=int,
                        default=[0, 0, 0],
                        help="""If calculating Cartesian moment integrals, \
                        specify the powers of the [x, y, z] operators.""")
    parser.add_argument('--L',
                        action='store_true',
                        help="""Calculate angular momentum integrals.""")
    parser.add_argument('--L_from_M',
                        action='store_true',
                        help="""Calculate angular momentum integrals, not from \
                        a recursion scheme, but from differentiating first \
                        moment integrals.""")
    parser.add_argument('--E',
                        action='store_true',
                        help="""Calculate electric field integrals.""")
    parser.add_argument('--EF_from_V',
                        action='store_true',
                        help="""Calculate electric field integrals, not from \
                        a recursion scheme, but from differentiating nuclear \
                        attraction integrals.""")
    parser.add_argument('--EFG_from_EF_from_V',
                        action='store_true',
                        help="""Calculate electric field gradient integrals, \
                        not from a recursion scheme, but from differentiating \
                        electric field integrals.""")
    parser.add_argument('--order-E',
                        nargs=3,
                        type=int,
                        default=[0, 0, 0],
                        help="""If calculating electric field, electric field \
                        gradient, or higher-order derivatives of the nuclear \
                        attraction integrals, specify the orders of the \
                        derivatives.""")
    parser.add_argument('--J',
                        action='store_true',
                        help="""Calculate 1-electron spatial spin-orbit \
                        integrals.""")
    parser.add_argument('--J_KF',
                        action='store_true',
                        help="""Calculate 1-electron spatial spin-orbit \
                        integrals, as described by King and Furlani, from \
                        differentiating nuclear attraction integrals.""")

    # Two-electron integrals
    parser.add_argument('--ERI',
                        action='store_true',
                        help="""Calculate electron repulsion integrals.""")
    parser.add_argument('--J2_KF',
                        action='store_true',
                        help="""Calculate 2-electron spatial spin-orbit \
                        integrals, as described by King and Furlani, from \
                        differentiating electron repulsion integrals.""")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = getargs()
