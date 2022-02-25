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
    PyArrayObject *array, *params;
    double x0, y0, x_sigma, y_sigma;
    int i, j, ny, nx;

    if (!PyArg_ParseTuple(
             args,
             "OO",
             &array, &params))
             //&y0, &x0, &y_sigma, &x_sigma, &height, &background))
        return NULL;

    ny = (int)PyArray_DIM(array,0);
    nx = (int)PyArray_DIM(array,1);

    y0 = INDd(params,0);
    x0 = INDd(params,1);
    y_sigma = INDd(params,2);
    x_sigma = INDd(params,3);
    //height = INDd(params,4);
    //background = INDd(params,5);

    for(j=0; j<ny; j++)
        for(i=0; i<nx; i++)
            IND2d(array,j,i) =
                INDd(params,5)
                + INDd(params,4) * exp(-0.5*(
                    pow((j-y0)/y_sigma, 2.0) + pow((i-x0)/x_sigma, 2.0)));

    return Py_BuildValue("");

}


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
