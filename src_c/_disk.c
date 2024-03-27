// Copyright (c) 2021-2024 Patricio Cubillos
// puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>

#include "ind.h"
#include "cdisk.h"

/* Function's doc string */
PyDoc_STRVAR(disk__doc__,
"Compute a disk image of given size, with True/False values for those\n\
pixels whose center lies closer/farther than radius from center.     \n\
                                                                     \n\
Parameters                                                           \n\
----------                                                           \n\
radius: Float                                                        \n\
   Disk radius in pixels.                                            \n\
center: 1D float ndarray                                             \n\
   Disk (y,x) center location.                                       \n\
size: 1D integer ndarray                                             \n\
   Output image (y,x) size in pixels.                                \n\
                                                                     \n\
Returns                                                              \n\
-------                                                              \n\
disk: 2D bool ndarray                                                \n\
   Disk image.                                                       \n\
status: Integer                                                      \n\
   Flag with 1/0 if any/none part of the disk lies outside of the    \n\
   image boundaries.                                                 \n\
ndisk: Integer                                                       \n\
   Number of pixels inside the disk.                                 \n\
                                                                     \n\
Examples                                                             \n\
--------                                                             \n\
>>> import disk as d                                                 \n\
>>> disk, s, n = d.disk(3.5, np.array([4.0,6.0]), np.array([8,8]))   \n\
>>> print(disk)                                                      \n\
>>> print(s, n)");


static PyObject *disk(PyObject *self, PyObject *args){
    double radius;
    PyArrayObject *d, *center, *size;
    npy_intp dims[2];
    int status, ndisk;

    if (!PyArg_ParseTuple(args, "dOO", &radius, &center, &size))
        return NULL;

    /* Allocate output array: */
    dims[0] = INDi(size,0);
    dims[1] = INDi(size,1);
    d = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);
    /* Compute disk mask:     */
    cdisk(
        d, radius, INDd(center,0), INDd(center,1),
        dims[0], dims[1],
        &status, &ndisk);

    return Py_BuildValue("Nii", d, status, ndisk);
}


/* Module's doc string */
PyDoc_STRVAR(disk_mod__doc__, "Circular disk image.");


static PyMethodDef disk_methods[] = {
    {"disk", disk, METH_VARARGS, disk__doc__},
    {NULL, NULL, 0, NULL}
};

/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_disk",
    disk_mod__doc__,
    -1,
    disk_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__disk (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
