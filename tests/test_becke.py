"""Becke tests files."""
from unittest import TestCase

from becke import compute_becke_weights, compute_select_becke

import numpy as np
from numpy.testing import assert_allclose


class TestBecke(TestCase):
    """Becke weight class tests."""

    def test_becke_sum2_one(self):
        """Test becke weights add up to one."""
        npoint = 100
        points = np.random.uniform(-5, 5, (npoint, 3))

        radii = np.array([0.5, 0.8])
        centers = np.array([[1.2, 2.3, 0.1], [-0.4, 0.0, -2.2]])
        weights0 = compute_select_becke(points, centers, radii, select=0, order=3)
        weights1 = compute_select_becke(points, centers, radii, select=1, order=3)

        assert_allclose(np.asarray(weights0) + np.asarray(weights1), np.ones(100))

    def test_becke_sum3_one(self):
        """Test becke weights add up to one with three centers."""
        npoint = 100
        points = np.random.uniform(-5, 5, (npoint, 3))

        radii = np.array([0.5, 0.8, 5.0])
        centers = np.array([[1.2, 2.3, 0.1], [-0.4, 0.0, -2.2], [2.2, -1.5, 0.0]])
        weights0 = compute_select_becke(points, centers, radii, select=0, order=3)
        weights1 = compute_select_becke(points, centers, radii, select=1, order=3)
        weights2 = compute_select_becke(points, centers, radii, select=2, order=3)

        assert_allclose(
            np.asarray(weights0) + np.asarray(weights1) + np.asarray(weights2),
            np.ones(100),
        )

    def test_becke_special_points(self):
        """Test becke weights for special cases."""
        radii = np.array([0.5, 0.8, 5.0])
        centers = np.array([[1.2, 2.3, 0.1], [-0.4, 0.0, -2.2], [2.2, -1.5, 0.0]])

        weights = compute_select_becke(centers, centers, radii, select=0, order=3)
        assert_allclose(np.asarray(weights), [1, 0, 0])

        weights = compute_select_becke(centers, centers, radii, select=1, order=3)
        assert_allclose(np.asarray(weights), [0, 1, 0])

        weights = compute_select_becke(centers, centers, radii, select=2, order=3)
        assert_allclose(np.asarray(weights), [0, 0, 1])

        # each point in seperate sectors.
        weights = compute_becke_weights(
            centers, centers, radii, np.array([0, 1, 2]), pt_ind=np.array([0, 1, 2, 3])
        )
        assert_allclose(np.asarray(weights), [1, 1, 1])

        weights = compute_becke_weights(
            centers,
            centers,
            radii,
            selects=np.array([0, 1]),
            pt_ind=np.array([0, 1, 3]),
        )
        assert_allclose(np.asarray(weights), [1, 1, 0])

        weights = compute_becke_weights(
            centers,
            centers,
            radii,
            selects=np.array([2, 0]),
            pt_ind=np.array([0, 2, 3]),
        )
        assert_allclose(np.asarray(weights), [0, 0, 0])

