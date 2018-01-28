#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include "ind.h"


PyDoc_STRVAR(linramp__doc__,
"Linear polynomial model:                                         \n\
   y(x) = a*(x-x0) + b                                            \n\
                                                                  \n\
Parameters                                                        \n\
----------                                                        \n\
params: 1D float ndarray                                          \n\
   Model slope (a), constant (b), and reference point (x0).       \n\
x: 1D float ndarray                                               \n\
   Evaluation points.                                             \n\
                                                                  \n\
Returns                                                           \n\
-------                                                           \n\
ramp: 1D float ndarray                                            \n\
   Linear polynomial ramp.");


static PyObject *linramp(PyObject *self, PyObject *args){
  PyArrayObject *x, *ramp, *params;
  double a, b, x0;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &x))
    return NULL;

  a  = INDd(params,0);
  b  = INDd(params,1);
  x0 = INDd(params,2);

  dims[0] = (int)PyArray_DIM(x,0);
  ramp = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  for(i=0; i<dims[0]; i++)
    INDd(ramp,i) = a*(INDd(x,i)-x0) + b;

  return Py_BuildValue("N", ramp);
}


/* Module's doc string */
PyDoc_STRVAR(linrampmod__doc__, "1D linear function.");


static PyMethodDef linramp_methods[] = {
  {"linramp", linramp, METH_VARARGS, linramp__doc__},
  {NULL,      NULL,    0,            NULL}
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "linramp", linrampmod__doc__, -1, linramp_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__linramp (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_linramp(void){
  Py_InitModule3("linramp", linramp_methods, linrampmod__doc__);
  import_array();
}
#endif
