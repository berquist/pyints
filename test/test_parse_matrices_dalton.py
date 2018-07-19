"""Test the parsing of integral matrices from DALTON text output and binary files."""

# import json
import os.path
import pickle

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
