#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#include "ind.h"

/* Function's doc string */
PyDoc_STRVAR(least_asym__doc__,
"x,y,max_iterations -> iteration count at that point, up to max_iterations");

static PyObject *least_asym(PyObject *self, PyObject *args){
  PyArrayObject *data, *dis, *weights;
  int w_truth;

  if(!PyArg_ParseTuple(args,"OOOi", &data, &dis, &weights, &w_truth)){
    return NULL;
  }

  if (!PyArray_SAMESHAPE(data, dis)){
    PyErr_Format(PyExc_ValueError,
     "Shape of data array not equal to that of distance");
    return NULL;
  }
  
  int dis_i, dis_j, dis_ii, dis_jj, dis_dim0, dis_dim1;
  double r, weight_sum, sum_weight_mean, mu, core, asym, var;
  asym = 0;

  dis_dim0 = (int)PyArray_DIM(dis, 0);
  dis_dim1 = (int)PyArray_DIM(dis, 1);

  for   (dis_i=0; dis_i<dis_dim0; dis_i++){
    for (dis_j=0; dis_j<dis_dim1; dis_j++){
      r = IND2d(dis, dis_i, dis_j);
      weight_sum = 0;
      sum_weight_mean = 0;
      core = 0;
      mu   = 0;
      var  = 0;

      for   (dis_ii=0; dis_ii<dis_dim0; dis_ii++){
        for (dis_jj=0; dis_jj<dis_dim1; dis_jj++){
          if(IND2d(dis,dis_ii,dis_jj)==r){
            weight_sum      += IND2d(weights, dis_ii, dis_jj);
            sum_weight_mean += IND2d(weights, dis_ii, dis_jj) *
                               IND2d(data, dis_ii, dis_jj);
          }
        }
      }
      mu = sum_weight_mean/weight_sum;
      for   (dis_ii=0; dis_ii<dis_dim0; dis_ii++){
        for (dis_jj=0; dis_jj<dis_dim1; dis_jj++){

          if (IND2d(dis, dis_ii, dis_jj)==r){
            core += IND2d(weights, dis_ii, dis_jj) *
                   (IND2d(data, dis_ii, dis_jj) - mu) *
                   (IND2d(data, dis_ii, dis_jj) - mu);
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
PyDoc_STRVAR(least_asymmod__doc__,
"Least asymmetry module.");


static PyMethodDef least_asym_methods[] = {
    {"least_asym", least_asym, METH_VARARGS, least_asym__doc__},
    {NULL,        NULL,      0,          NULL}
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "least_asym", least_asymmod__doc__, -1, least_asym_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_least_asym (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void initleast_asym(void){
  Py_InitModule3("least_asym", least_asym_methods, least_asymmod__doc__);
  import_array();
}
#endif
