#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>

#include "ind.h"
#include "disk_c.h"

/* Function's doc string */
PyDoc_STRVAR(pydisk__doc__,
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
pydisk: 2D bool ndarray                                              \n\
   Disk image.                                                       \n\
status: Integer                                                      \n\
   Flag with 1/0 if any part of the disk lies outside/inside of the  \n\
   image boundaries.                                                 \n\
ndisk: Integer                                                       \n\
   Number of pixels inside the disk.                                 \n\
                                                                     \n\
Examples                                                             \n\
--------                                                             \n\
>>> import cdisk as d                                                \n\
>>> disk, s, n = d.pydisk(3.5, np.array([4.0,6.0]), np.array([8,8])) \n\
>>> print(disk)                                                      \n\
>>> print(s, n)");


static PyObject *pydisk(PyObject *self, PyObject *args){
  double radius;
  PyArrayObject *pydisk, *center, *size;
  npy_intp dims[2];
  int i, j, ny, nx, status, ndisk;
  char **d;

  if (!PyArg_ParseTuple(args, "dOO", &radius, &center, &size))
      return NULL;

  ny = dims[0] = INDi(size,0);
  nx = dims[1] = INDi(size,1);
  pydisk = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);

  /* Allocate disk: */
  d    = (char **)malloc(ny*    sizeof(char *));
  d[0] = (char  *)calloc(ny*nx, sizeof(char  ));
  for (i=1; i<ny; i++)
    d[i] = d[0] + nx*i;

  disk(d, radius, INDd(center,0), INDd(center,1), INDi(size,0), INDi(size,1),
       &status, &ndisk);

  for   (i=0; i<ny; i++)
    for (j=0; j<nx; j++)
      IND2b(pydisk,i,j) = d[i][j];

  free(d[0]);
  free(d);
  return Py_BuildValue("Nii", pydisk, status, ndisk);
}


/* Module's doc string */
PyDoc_STRVAR(cdisk__doc__, "Circular disk image.");


static PyMethodDef cdisk_methods[] = {
    {"pydisk", pydisk, METH_VARARGS, pydisk__doc__},
    {NULL,     NULL,   0,            NULL}
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "cdisk", cdisk__doc__, -1, cdisk_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_cdisk (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void initcdisk(void){
  Py_InitModule3("cdisk", cdisk_methods, cdisk__doc__);
  import_array();
}
#endif
