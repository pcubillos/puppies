# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

import os
import sys

import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__) + '/..') + '/'
sys.path.append(ROOT)
import puppies.stats as ps


def test_medstd():
    data = np.array([1,3,4,5,6,7,7])
    std, med = ps.medstd(data, median=True)
    assert med == 5.0
    assert std == 2.23606797749979


def test_medstd_masked():
    data = np.array([1,3,4,5,6,7,7])
    mask = np.array([1,1,1,0,0,0,0], bool)
    std, med = ps.medstd(data, mask, median=True)
    assert std == 1.5811388300841898
    assert med == 3.0


def test_medstd_2D():
    b = np.array([
        [1, 3, 4,  5, 6,  7, 7],
        [4, 3, 4, 15, 6, 17, 7],
        [9, 8, 7,  6, 5,  4, 3]])
    data = np.array([b, 1-b, 2+b])
    std, med = ps.medstd(data, median=True, axis=2)
    np.testing.assert_equal(med, np.median(data, axis=2))
    np.testing.assert_allclose(std, np.array([
        [2.23606798, 6.05530071, 2.1602469 ],
        [2.23606798, 6.05530071, 2.1602469 ],
        [2.23606798, 6.05530071, 2.1602469 ]]))


def test_sigrej():
    x = np.array([65., 667, 84, 968, 70, 66, 78, 47, 71, 56, 65, 60])
    mask, mean, std, median, medstd = ps.sigrej(
        x, [2,1], retmean=True, retstd=True, retmedian=True, retmedstd=True)
    np.testing.assert_equal(mask, np.array(
        [ True, False,  True, False,  True,  True,  True,  True,  True,
          True,  True,  True]))
    assert mean == 66.2
    assert std == 10.58090523327544
    assert median == 65.5
    assert medstd == 10.606601717798213

