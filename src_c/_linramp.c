// Copyright (c) 2021-2024 Patricio Cubillos
// puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include "ind.h"


PyDoc_STRVAR(linramp__doc__,
"Linear polynomial model:                                         \n\
   ramp(t) = r1*(t-t0) + r0                                       \n\
                                                                  \n\
Parameters                                                        \n\
----------                                                        \n\
params: 1D float ndarray                                          \n\
   Model slope (r1), constant (r0), and reference point (t0).     \n\
t: 1D float ndarray                                               \n\
   Evaluation points.                                             \n\
                                                                  \n\
Returns                                                           \n\
-------                                                           \n\
ramp: 1D float ndarray                                            \n\
   Linear polynomial ramp.");


static PyObject *linramp(PyObject *self, PyObject *args){
    PyArrayObject *t, *ramp, *params;
    double r1, r0, t0;
    int i;
    npy_intp dims[1];

    if(!PyArg_ParseTuple(args, "OO", &params, &t))
        return NULL;

    r1 = INDd(params,0);
    r0 = INDd(params,1);
    t0 = INDd(params,2);

    dims[0] = (int)PyArray_DIM(t,0);
    ramp = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    for(i=0; i<dims[0]; i++)
        INDd(ramp,i) = r1*(INDd(t,i)-t0) + r0;

    return Py_BuildValue("N", ramp);
}


/* Module's doc string */
PyDoc_STRVAR(
    linramp_mod__doc__,
    "1D linear function.");


static PyMethodDef linramp_methods[] = {
    {"linramp", linramp, METH_VARARGS, linramp__doc__},
    {NULL, NULL, 0, NULL}
};


/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_linramp",
    linramp_mod__doc__,
    -1,
    linramp_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__linramp (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
