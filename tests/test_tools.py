# Copyright (c) 2021 Patricio Cubillos
# puppies is open-source software under the MIT license (see LICENSE)

import os
import sys
import pytest

import astropy.units as u

ROOT = os.path.realpath(os.path.dirname(__file__) + '/..') + '/'
sys.path.append(ROOT)
import puppies.tools as pt


@pytest.mark.skip(reason="To be replaced by mc3.Log")
def test_msg():
    pass


@pytest.mark.skip(reason="To be replaced by mc3.Log")
def test_warning():
    pass


@pytest.mark.skip(reason="To be replaced by mc3.Log")
def test_error():
    pass


def test_getpar_scalar():
    value, uncert = pt.getpar('1.0')
    assert value  == 1.0
    assert uncert == 0.0


def test_getpar_quantity():
    value, uncert = pt.getpar('2.0 cm')
    assert value  == 2.0 * u.Unit('cm')
    assert uncert == 0.0 * u.Unit('cm')


def test_getpar_uncert():
    value, uncert = pt.getpar('3.0 0.5 cm')
    assert value  == 3.0 * u.Unit('cm')
    assert uncert == 0.5 * u.Unit('cm')


@pytest.mark.skip(reason="This is numpy's concatenate")
def test_cat():
    pass


@pytest.mark.skip(reason='TBI')
def test_parse_model():
    pass


@pytest.mark.skip(reason='TBI')
def test_parray():
    pass


@pytest.mark.skip(reason='TBI')
def test_newparams():
    pass


@pytest.mark.skip(reason='TBI')
def test_loadparams():
    pass


@pytest.mark.skip(reason='TBI')
def test_saveparams():
    pass
