#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>
#include<math.h>

#include "ind.h"


PyDoc_STRVAR(mandelecl__doc__,
"Secondary-eclipse light-curve model from Mandel & Agol (2002).\n\
                                                               \n\
Parameters                                                     \n\
----------                                                     \n\
params: 1D float ndarray                                       \n\
   Eclipse model parameters:                                   \n\
     midpt:  Center of eclipse.                                \n\
     width:  Eclipse duration between contacts 1 to 4.         \n\
     depth:  Eclipse depth.                                    \n\
     t12:    Eclipse duration between contacts 1 to 2.         \n\
     t34:    Eclipse duration between contacts 3 to 4.         \n\
     flux:   Out-of-eclipse flux level.                        \n\
t: 1D float ndarray                                            \n\
   The lightcurve's phase/time points.                         \n\
                                                               \n\
Returns                                                        \n\
-------                                                        \n\
eclipse: 1D float ndarray                                      \n\
   Mandel & Agol eclipse model evaluated at points t.");

static PyObject *mandelecl(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, depth, t12, t34, flux;
  double t1, t2, t3, t4, p, z, k0, k1;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt = INDd(params, 0);
  width = INDd(params, 1);
  depth = INDd(params, 2);
  t12   = INDd(params, 3);
  t34   = INDd(params, 4);
  flux  = INDd(params, 5);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  if(depth == 0){
      for(i=0; i<dims[0]; i++){
        INDd(eclipse, i) = flux;
      }
      return Py_BuildValue("N", eclipse);
    }

  /* Time of contact points: */
  t1 = midpt - width/2;

  if ((t1+t12) < midpt)
    t2 = t1 + t12;
  else
    t2 = midpt;

  t4 = midpt + width/2;
  if ((t4-t34) > midpt)
    t3 = t4 - t34;
  else
    t3 = midpt;

  p = sqrt(fabs(depth)) * (depth/fabs(depth));

  for(i=0; i<dims[0]; i++){
    INDd(eclipse,i) = 1.0;
    if(INDd(t,i) >= t2  &&  INDd(t,i) <= t3){
      INDd(eclipse,i) = 1 - depth;
    }
    else if (p != 0){
      /* Use Mandel & agol (2002) for ingress of eclipse: */
      if (INDd(t,i) > t1  &&  INDd(t,i) < t2){
        z  = -2*p*(INDd(t,i)-t1)/t12 + 1 + p;
        k0 = acos((p*p+z*z-1)/2/p/z);
        k1 = acos((1-p*p+z*z)/2/z);
        INDd(eclipse,i) = 1 - depth/fabs(depth)/M_PI * (p*p*k0 + k1
                                    - sqrt((4*z*z - pow((1+z*z-p*p),2))/4));
      }
      else if (INDd(t,i) > t3  &&  INDd(t,i) < t4){
        z  = 2*p*(INDd(t,i)-t3)/t34 + 1 - p;
        k0 = acos((p*p+z*z-1)/2/p/z);
        k1 = acos((1-p*p+z*z)/2/z);
        INDd(eclipse,i) = 1-depth/fabs(depth)/M_PI*(p*p*k0 + k1
                                    - sqrt((4*z*z - pow((1+z*z-p*p),2))/4));
      }
    }
    INDd(eclipse,i) *= flux;
  }

  return Py_BuildValue("N", eclipse);
}


PyDoc_STRVAR(mandeleclmod__doc__,
             "Mandel-Agol eclipse light-curve model.\n");


static PyMethodDef mandelecl_methods[] = {
  {"mandelecl", mandelecl, METH_VARARGS, mandelecl__doc__},
  {NULL,        NULL,      0,            NULL}
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "mandelecl", mandeleclmod__doc__, -1, mandelecl_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__mandelecl (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_mandelecl(void){
  Py_InitModule3("mandelecl", mandelecl_methods, mandeleclmod__doc__);
  import_array();
}
#endif
