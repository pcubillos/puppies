#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>

#include "ind.h"

/* Function's doc string */
PyDoc_STRVAR(gauss2D__doc__,
"Compute a 2D Gaussian array.                                        \n\
                                                                     \n\
Parameters                                                           \n\
----------                                                           \n\
ny: Integer                                                          \n\
   Size of the first dimension of the 2D output array.               \n\
nx: Integer                                                          \n\
   Size of the second dimension of the 2D output array.              \n\
y0: Float                                                            \n\
   Center of the Gaussian along the first dimension with respect to  \n\
   the origin at the first pixel.                                    \n\
x0: Float                                                            \n\
   Center of the Gaussian along the second dimension with respect to \n\
   the origin at the first pixel.                                    \n\
sigmay: Float                                                        \n\
   Gaussian standard deviation along the first dimension.            \n\
sigmax: Float                                                        \n\
   Gaussian standard deviation along the second dimension.           \n\
height: Float                                                        \n\
   Value of the Gaussian at (y0,x0).  If left as zero, set the height\n\
   such that the integral of the array equals one.                   \n\
background: Float                                                    \n\
   Constant added value added to each pixel in the output array      \n\
   (zero by default).                                                \n\
                                                                     \n\
Returns                                                              \n\
-------                                                              \n\
mat: 2D float ndarray                                                \n\
   A 2D Gaussian array of shape (ny,nx).");


/* The wrapper to the underlying C function */
static PyObject *gauss2D(PyObject *self, PyObject *args){
  double x0, y0, sigmax, sigmay, background=0, height=0.0;
  PyArrayObject *mat;
  npy_intp dims[2];
  int i, j, nx, ny;

  if (!PyArg_ParseTuple(args, "iidddd|dd", &ny, &nx, &y0, &x0, &sigmay,
                                           &sigmax, &height, &background))
      return NULL;

  if (height == 0.0)
    height = 1.0 / (2.0 * 3.141592653589793 * sigmay * sigmax);

  dims[0] = ny;
  dims[1] = nx;
  mat = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

  for   (j=0; j<ny; j++){
    for (i=0; i<nx; i++){
      IND2d(mat,j,i) = height * exp(-0.5*(pow((j-y0)/sigmay, 2.0) +
                                         pow((i-x0)/sigmax, 2.0))) + background;
    }
  }
  return Py_BuildValue("N", mat);
}


/* Module's doc string */
PyDoc_STRVAR(gauss__doc__, "2D Gaussian function.");


static PyMethodDef gauss_methods[] = {
    {"gauss2D", gauss2D, METH_VARARGS, gauss2D__doc__},
    {NULL,    NULL,  0,            NULL}
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "gauss", gauss__doc__, -1, gauss_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_gauss (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void initgauss(void){
  Py_InitModule3("gauss", gauss_methods, gauss__doc__);
  import_array();
}
#endif
