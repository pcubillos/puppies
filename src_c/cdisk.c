#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>

#include "ind.h"

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
   Flag with 1/0 if any part of the disk lies outside/inside of the  \n\
   image boundaries.                                                 \n\
ndisk: Integer                                                       \n\
   Number of pixels inside the disk.                                 \n\
                                                                     \n\
Examples                                                             \n\
--------                                                             \n\
>>> import cdisk as d                                                \n\
>>> disk, s, n = d.disk(3.5, np.array([4.0,6.0]), np.array([8,8]))   \n\
>>> print(disk)                                                      \n\
>>> print(s, n)");


static PyObject *disk(PyObject *self, PyObject *args){
  double radius;
  PyArrayObject *d, *center, *size;
  npy_intp dims[2];
  int i, j, ny, nx, status=0, ndisk, n=0;
  double yctr, xctr;

  if (!PyArg_ParseTuple(args, "dOO", &radius, &center, &size))
      return NULL;

  ny = dims[0] = INDi(size,0);
  nx = dims[1] = INDi(size,1);
  d = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);

  yctr = INDd(center,0);
  xctr = INDd(center,1);
  /* Alert if the center lies outside the image:                           */
  if ( (yctr-radius) < 0 || (yctr+radius) > (ny-1) ||
       (xctr-radius) < 0 || (xctr+radius) > (nx-1) )
    status = 1;

  for   (i=0; i<ny; i++)
    for (j=0; j<nx; j++){
      /* Is the point disk[i][j] inside the disk?                          */
      IND2b(d,i,j) = (i-yctr)*(i-yctr) + (j-xctr)*(j-xctr) <= radius*radius;
      n += IND2b(d,i,j);
    }
  /* Number of pixels within radius in ndisk:                              */
  ndisk = n;

  return Py_BuildValue("Nii", d, status, ndisk);
}


/* Module's doc string */
PyDoc_STRVAR(cdisk__doc__, "Circular disk image.");


static PyMethodDef cdisk_methods[] = {
    {"disk", disk, METH_VARARGS, disk__doc__},
    {NULL,   NULL, 0,            NULL}
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
