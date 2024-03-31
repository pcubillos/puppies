// Copyright (c) 2021-2024 Patricio Cubillos
// puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

/* Definitions for indexing Numpy arrays: */

/* 1D ndarray: */
#define INDd(a,i) *((double *)(PyArray_DATA(a) + i * PyArray_STRIDE(a, 0)))
#define INDi(a,i) *((int    *)(PyArray_DATA(a) + i * PyArray_STRIDE(a, 0)))
/* 2D ndarray: */
#define IND2d(a,i,j) *((double *)(PyArray_DATA(a) + i * PyArray_STRIDE(a, 0) \
                                                  + j * PyArray_STRIDE(a, 1)))
#define IND2i(a,i,j) *((int    *)(PyArray_DATA(a) + i * PyArray_STRIDE(a, 0) \
                                                  + j * PyArray_STRIDE(a, 1)))
#define IND2b(a,i,j) *((_Bool  *)(PyArray_DATA(a) + i * PyArray_STRIDE(a, 0) \
                                                  + j * PyArray_STRIDE(a, 1)))
