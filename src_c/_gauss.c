// Copyright (c) 2021 Patricio Cubillos
// puppies is open-source software under the MIT license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include "ind.h"


PyDoc_STRVAR(gauss2D__doc__,
"Exponential model.                                                 \n\
                                                                    \n\
Parameters                                                          \n\
----------                                                          \n\
params: 1D float ndarray                                            \n\
   Model parameters:                                                \n\
     goal: goal as t tends to infinity.                             \n\
     r1: exponential rate.                                          \n\
     r0: exponential time offset.                                   \n\
     pm: Set to +/- 1.0 to get a decaying/rising exponential.       \n\
t: 1D float ndarray                                                 \n\
   Array of time/phase points.                                      \n\
                                                                    \n\
Returns                                                             \n\
-------                                                             \n\
ramp: 1D float ndarray                                              \n\
   Exponential ramp.");


static PyObject *gauss2D(PyObject *self, PyObject *args){
    PyArrayObject
        *array, *t, *params;
    double background, r0, r1, pm;
    double
        height=0.0,
        //background=0.0,
        x0, y0, x_sigma, y_sigma;
    int i, j, ny, nx;
    npy_intp dims[2];

    if (!PyArg_ParseTuple(args, "OO", &params, &t))
        return NULL;

    background = INDd(params,0);
    r1 = INDd(params,1);
    r0 = INDd(params,2);
    pm = INDd(params,3);

    dims[0] = (int)PyArray_DIM(t,0);
    dims[1] = (int)PyArray_DIM(t,0);
    array = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    //for(j=0; j<ny; j++)
    //    for(i=0; i<nx; i++)
    //        IND2d(array,j,i) =
    //            background
    //            + exp(-0.5 * pow((j-y0)/y_sigma, 2.0));

    for(i=0; i<dims[0]; i++)
        IND2d(array,i,i) =
            background
            + pm*exp(-r1*INDd(t,i) + r0);

    return Py_BuildValue("N", array);

}


/* Module's doc string */
PyDoc_STRVAR(
    gauss_mod__doc__,
    "2D Gaussian function.");


static PyMethodDef gauss_mod_methods[] = {
    {"gauss2D", gauss2D, METH_VARARGS, gauss2D__doc__},
    {NULL, NULL, 0, NULL}
};


/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_gauss",
    gauss_mod__doc__,
    -1,
    gauss_mod_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__gauss (void){
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
