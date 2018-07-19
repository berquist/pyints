"""Test the parsing of integral matrices from DALTON text output and binary files."""

import os.path

from pyints import parse_matrices_dalton


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
    return
