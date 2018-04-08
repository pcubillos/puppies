#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>
#include<math.h>

#include "ind.h"


PyDoc_STRVAR(quadramp__doc__,
"Quadratic polynomial model:                                    \n\
   y(t) = r2*(t-t0)**2 + r1*(t-t0) + r0                         \n\
                                                                \n\
Parameters                                                      \n\
----------                                                      \n\
params: 1D float ndarray                                        \n\
   Model quadratic (r2), linear (r1), constant (r0), and reference \n\
   point (t0).                                                  \n\
t: 1D float ndarray                                             \n\
   Evaluation points.                                           \n\
                                                                \n\
Returns                                                         \n\
-------                                                         \n\
ramp: 1D float ndarray                                          \n\
   Quadratic polynomial ramp.");


static PyObject *quadramp(PyObject *self, PyObject *args){
  PyArrayObject *t, *ramp, *params;
  double r0, r1, r2, t0;
  int i;
  npy_intp dims[1];

  if (!PyArg_ParseTuple(args, "OO", &params, &t))
      return NULL;

  r2 = INDd(params,0);
  r1 = INDd(params,1);
  r0 = INDd(params,2);
  t0 = INDd(params,3);

  dims[0] = (int)PyArray_DIM(t, 0);

  ramp = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  for (i=0; i<dims[0]; i++)
    INDd(ramp,i) = r2*pow((INDd(t,i) - t0), 2) + r1*(INDd(t,i) - t0) + r0;

  return Py_BuildValue("N", ramp);
}


/* Module's doc string */
PyDoc_STRVAR(quadrampmod__doc__, "1D quadratic function.");


static PyMethodDef quadramp_methods[] = {
  {"quadramp", quadramp, METH_VARARGS, quadramp__doc__},
  {NULL,       NULL,     0,            NULL}
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "quadramp", quadrampmod__doc__, -1, quadramp_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__quadramp (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_quadramp(void){
  Py_InitModule3("quadramp", quadramp_methods, quadrampmod__doc__);
  import_array();
}
#endif
