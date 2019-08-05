"""Cython extension forb becke weights."""

from libc.math cimport sqrt
import numpy as np
cimport numpy as np

cdef double distance(double[:] pt1, double[:] pt2):
    cdef int i, n
    cdef double total = 0.
    n = pt1.shape[0]
    for i in range(n):
        total += (pt1[i] - pt2[i]) ** 2
    return sqrt(total)

cpdef double[:] compute_select_becke(double[:, :] points,
                                     double[:, :] atoms,
                                     double[:] radii,
                                     int select,
                                     int order):
    cdef int i, j, k, l, m, n
    cdef double nom, denom, p
    cdef double u_ab, s
    m = atoms.shape[0]  # number of atoms
    n = points.shape[0]   # number of points
    cdef double[:, :] alpha = np.zeros((m, m), dtype=float)
    # compute alpha
    for i in range(m):
        for j in range(i + 1, m):
            u_ab = (radii[i] - radii[j]) / (radii[i] + radii[j])
            u_ab /= (u_ab**2 - 1)
            if u_ab > 0.45:
                u_ab = 0.45
            elif u_ab < -0.45:
                u_ab = -0.45
            alpha[i][j] = u_ab
            alpha[j][i] = -alpha[i][j]
    # compute atoms dis
    cdef double[:, :] at_dists = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            at_dists[i][j] = distance(atoms[i], atoms[j])
            at_dists[j][i] = at_dists[i][j]
    # actual compute becke weights
    cdef double[:] becke_wts = np.ones(n, dtype=float)
    # loop over each points
    for i in range(n):
        # loop over atom 1
        nom = 0.
        denom = 0.
        for j in range(m):
            p = 1.
            # loop over atom2
            for k in range(m):
                if j == k:
                    continue
                miu = (distance(points[i], atoms[j]) -
                       distance(points[i], atoms[k])) / at_dists[j][k]
                nu = miu + alpha[j][k] * (1 - miu**2)
                for l in range(order):
                    nu = 0.5 * nu * (3 - nu**2)
                s = 0.5 * (1 - nu)
                p *= s
            # add each jk pari
            denom += p
            if j == select:
                nom = p
        becke_wts[i] *= nom / denom
    return becke_wts

cpdef np.ndarray[double, ndim=1] compute_becke_weights(double[:, :] points,
                                                       double[:, :] atoms,
                                                       double[:] radii,
                                                       np.int_t[:] selects,
                                                       np.int_t[:] pt_ind,
                                                       int order=3):
    cdef int i, m, n, ind1, ind2
    m = selects.shape[0]
    n = points.shape[0]
    cdef np.ndarray tot_wts = np.zeros(n, dtype=float)
    # selected atoms
    for i in range(m):
        ind1 = pt_ind[i]
        ind2 = pt_ind[i + 1]
        tot_wts[ind1:ind2] = compute_select_becke(points[ind1:ind2],
                                                  atoms,
                                                  radii,
                                                  selects[i],
                                                  order)
    return tot_wts
