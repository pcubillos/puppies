// Copyright (c) 2021-2024 Patricio Cubillos
// puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#include "ind.h"

/* Function's doc string */
PyDoc_STRVAR(asymmetry__doc__,
"Compute asymmetry of a 2D array\n\
                         \n\
Parameters               \n\
----------               \n\
data: 2D float ndarray   \n\
dis: 2D float ndarray    \n\
weights: 2D float ndarray\n\
w_truth: Integer         \n\
");

static PyObject *asymmetry(PyObject *self, PyObject *args){
    PyArrayObject *data, *dis, *weights;
    int w_truth, i, j, k, l, dis_dim0, dis_dim1;
    double r, weight_sum, sum_weight_mean, mu, core, asym, var;
    asym = 0;

    if(!PyArg_ParseTuple(args,"OOOi", &data, &dis, &weights, &w_truth)){
        return NULL;
    }

    if (!PyArray_SAMESHAPE(data, dis)){
        PyErr_Format(
            PyExc_ValueError,
            "Shape of data array does not equal that of distance.");
        return NULL;
    }

    dis_dim0 = (int)PyArray_DIM(dis, 0);
    dis_dim1 = (int)PyArray_DIM(dis, 1);

    for (i=0; i<dis_dim0; i++){
        for (j=0; j<dis_dim1; j++){
            r = IND2d(dis, i, j);
            weight_sum = 0;
            sum_weight_mean = 0;
            core = 0;
            mu   = 0;
            var  = 0;

            for (k=0; k<dis_dim0; k++){
                for (l=0; l<dis_dim1; l++){
                    if(IND2d(dis, k, l)==r){
                      weight_sum += IND2d(weights, k, l);
                      sum_weight_mean +=
                          IND2d(weights, k, l) * IND2d(data, k, l);
                    }
                }
            }
            mu = sum_weight_mean/weight_sum;
            for (k=0; k<dis_dim0; k++){
                for (l=0; l<dis_dim1; l++){
                    if (IND2d(dis, k, l)==r){
                        core +=
                            IND2d(weights,k,l) *
                            (IND2d(data,k,l) - mu) * (IND2d(data,k,l) - mu);
                    }
                }
            }
            var = core/weight_sum;
            if (w_truth == 1){
              var = var*var;
            }
            asym += var;
        }
    }
    return Py_BuildValue("d", asym);
}

/* Module's doc string */
PyDoc_STRVAR(
    asymmetry_mod__doc__,
    "Least asymmetry C module.");


static PyMethodDef asymmetry_methods[] = {
    {"asymmetry", asymmetry, METH_VARARGS, asymmetry__doc__},
    {NULL, NULL, 0, NULL}
};

/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_asymmetry",
    asymmetry_mod__doc__,
    -1,
    asymmetry_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__asymmetry (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
