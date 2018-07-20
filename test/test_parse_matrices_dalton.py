"""Test the parsing of integral matrices from DALTON text output and binary files."""

# import json
import os.path
import pickle
import yaml

import numpy as np

from pyints import parse_matrices_dalton


__filedir__ = os.path.realpath(os.path.dirname(__file__))
refdir = os.path.join(__filedir__, 'reference_data')


def test_parse_element_dalton():
    """Test parsing Fortran floating-point matrix elements using a
    regex.
    """
    element_y = "         -0.157859751054D-01"
    ref_y = -0.157859751054e-01
    element_n = "-0.02226873"
    ref_n = -0.02226873
    thresh = 1.0e-8
    res_y = parse_matrices_dalton.parse_element_dalton(element_y)
    assert abs(ref_y - res_y) < thresh
    res_n = parse_matrices_dalton.parse_element_dalton(element_n)
    assert abs(ref_n - res_n) < thresh


def test_parse_matrix_dalton():
    """Test parsing a single sparse matrix from a DALTON output file."""
    # outputfilename = os.path.join(refdir, 'LiH_STO-3G', 'LiH_STO-3G_dalton.out')
    testcase = 'LiH_STO-3G'
    stub = 'dalton_matrix'
    matfilename = os.path.join(refdir, testcase, '{stub}.dat'.format(stub=stub))
    # jsonfilename = os.path.join(refdir, testcase, '{stub}.json'.format(stub=stub))
    picklefilename = os.path.join(refdir, testcase, '{stub}.pickle'.format(stub=stub))
    assert os.path.exists(matfilename)
    with open(matfilename) as matfile:
        res_spmat = parse_matrices_dalton.parse_matrix_dalton(matfile)
    ## TODO JSON would be better than a pickle, but won't work out of
    ## the box, because all keys are converted to strings on
    ## serialization, so a custom deserializer/JSONDecoder must be
    ## written.
    # assert os.path.exists(jsonfilename)
    # with open(jsonfilename) as jsonfile:
    #     ref_spmat = json.load(jsonfile)
    assert os.path.exists(picklefilename)
    with open(picklefilename, 'rb') as picklefile:
        ref_spmat = pickle.load(picklefile)
    assert res_spmat == ref_spmat


def test_sparse_to_dense_matrix_dalton():
    """Test converting a sparse (dictionary of dictionary of floats)
    matrix to a dense (NumPy) matrix.
    """
    testcase = 'LiH_STO-3G'
    stub_source = 'dalton_matrix'
    stub_ref = 'dalton_matrix'
    sourcefilename = os.path.join(refdir, testcase, '{stub}.pickle'.format(stub=stub_source))
    reffilename = os.path.join(refdir, testcase, '{stub}.npy'.format(stub=stub_ref))
    with open(sourcefilename, 'rb') as sourcefile:
        spmat = pickle.load(sourcefile)
    # TODO this is an assumption
    dim = max(spmat.keys())
    res_mat = parse_matrices_dalton.sparse_to_dense_matrix_dalton(spmat, dim)
    ref_mat = np.load(reffilename)
    np.testing.assert_allclose(res_mat, ref_mat, rtol=0, atol=1.0e-14)


def test_parse_spin_orbit_2el():
    testcase = 'LiH_STO-3G'
    stub_matfile = 'dalton_spin_orbit_2el_reversed'
    stub_yamlfile = 'dalton_spin_orbit_2el'
    matfilename = os.path.join(refdir, testcase, '{stub}.dat'.format(stub=stub_matfile))
    yamlfilename = os.path.join(refdir, testcase, '{stub}.yaml'.format(stub=stub_yamlfile))
    with open(matfilename) as matfile:
        for line in matfile:
            if 'spin-orbit two-electron integrals and' in line:
                # nlines = int(line.split()[0])
                while '##' not in line:
                    line = next(matfile)
                # Assume that the largest index printed is the
                # dimension/number of AOs.
                dim = max([int(x) for x in line.split()[1:4]])
    with open(matfilename) as matfile:
        res_so2el = parse_matrices_dalton.parse_spin_orbit_2el(matfile, dim)
    with open(yamlfilename) as yamlfile:
        ref_so2el = yaml.load(yamlfile)
    for indices, coord, val in ref_so2el:
        indices = tuple(np.array(indices, dtype=int) - 1)
        assert res_so2el[coord][indices] == val
    # Make sure the elements above are the only ones that are nonzero.
    x, y, z = res_so2el
    l = x[np.abs(x) > 0].tolist() \
        + y[np.abs(y) > 0].tolist() \
        + z[np.abs(z) > 0].tolist()
    assert len(l) == len(ref_so2el)


if __name__ == '__main__':
    test_parse_element_dalton()
    test_parse_matrix_dalton()
    test_sparse_to_dense_matrix_dalton()
    test_parse_spin_orbit_2el()
