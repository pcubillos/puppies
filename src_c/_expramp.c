#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>
#include<math.h>

#include "ind.h"

PyDoc_STRVAR(expramp__doc__,
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


static PyObject *expramp(PyObject *self, PyObject *args){
  PyArrayObject *t, *ramp, *params;
  double goal, r0, r1, pm;
  int i;
  npy_intp dims[1];

  if (!PyArg_ParseTuple(args, "OO", &params, &t))
    return NULL;

  goal = INDd(params,0);
  r1   = INDd(params,1);
  r0   = INDd(params,2);
  pm   = INDd(params,3);

  dims[0] = (int)PyArray_DIM(t,0);
  ramp = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  for(i=0; i<dims[0]; i++)
    INDd(ramp,i) = goal + pm*exp(-r1*INDd(t,i) + r0);

  return Py_BuildValue("N", ramp);
}


/* Module's doc string */
PyDoc_STRVAR(exprampmod__doc__, "1D rising exponential function.");


static PyMethodDef expramp_methods[] = {
  {"expramp", expramp, METH_VARARGS, expramp__doc__},
  {NULL,      NULL,    0,            NULL}
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "expramp", exprampmod__doc__, -1, expramp_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__expramp (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_expramp(void){
  Py_InitModule3("expramp", expramp_methods, exprampmod__doc__);
  import_array();
}
#endif
