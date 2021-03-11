// Copyright (c) 2021 Patricio Cubillos
// puppies is open-source software under the MIT license (see LICENSE)

#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>
#include<math.h>

#include "ind.h"


double sign(double x){
  if (x<0)
    return -1.0;
  return 1.0;
}

double k0(double p, double z){
  if (p==0)
    return 0.0;
  return acos(0.5*(p*p + z*z - 1)/(p*z));
}

double k1(double p, double z){
  return acos(0.5*(1 - p*p + z*z)/z);
}


PyDoc_STRVAR(mandelecl__doc__,
"Secondary-eclipse light-curve model from Mandel & Agol (2002).\n\
                                                               \n\
Parameters                                                     \n\
----------                                                     \n\
params: 1D float ndarray                                       \n\
   Eclipse model parameters:                                   \n\
     midpt:  Center of eclipse.                                \n\
     width:  Eclipse duration between contacts 1 and 4.        \n\
     depth:  Eclipse depth.                                    \n\
     ting:   Eclipse duration between contacts 1 and 2.        \n\
     tegr:   Eclipse duration between contacts 3 and 4.        \n\
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
  double midpt, width, depth, ting, tegr, flux;
  double t1, t2, t3, t4, p, z;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt = INDd(params, 0);
  width = INDd(params, 1);
  depth = INDd(params, 2);
  ting  = INDd(params, 3);
  tegr  = INDd(params, 4);
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

  if ((t1+ting) < midpt)
    t2 = t1 + ting;
  else
    t2 = midpt;

  t4 = midpt + width/2;
  if ((t4-tegr) > midpt)
    t3 = t4 - tegr;
  else
    t3 = midpt;

  p = sign(depth) * sqrt(fabs(depth));

  for(i=0; i<dims[0]; i++){
    INDd(eclipse,i) = 1.0;
    /* Out of eclipse: */
    if (INDd(t,i) < t1 || INDd(t,i) > t4){
      INDd(eclipse,i) += depth;
    }
    /* Totality:       */
    else if (INDd(t,i) >= t2  &&  INDd(t,i) <= t3){
    }
    /* Eq. (1) of Mandel & Agol (2002) for ingress/egress:  */
    else if (p != 0){
      if (INDd(t,i) > t1  &&  INDd(t,i) < t2){
        z  = -2*p*(INDd(t,i)-t1)/ting + 1 + p;
      }
      else{ /* (INDd(t,i) > t3  &&  INDd(t,i) < t4)         */
        z  =  2*p*(INDd(t,i)-t3)/tegr + 1 - p;
      }
      INDd(eclipse,i) += depth - sign(depth)/M_PI * (p*p*k0(p,z) + k1(p,z)
                                      - sqrt(z*z - 0.25*pow(1+z*z-p*p,2)));
    }
    INDd(eclipse,i) *= flux;
  }

  return Py_BuildValue("N", eclipse);
}


PyDoc_STRVAR(eclipse_flat__doc__,
"Secondary-eclipse light-curve model with flat baseline and\n\
independent ingress and egress depths.");

static PyObject *eclipse_flat(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, idepth, edepth, ting, tegr, flux;
  double t1, t2, t3, t4, pi, pe, z;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt  = INDd(params, 0);
  width  = INDd(params, 1);
  idepth = INDd(params, 2);  // Ingress depth
  edepth = INDd(params, 3);  // Egress depth
  ting   = INDd(params, 4);
  tegr   = INDd(params, 5);
  flux   = INDd(params, 6);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  /* Time of contact points: */
  t1 = midpt - width/2;
  t2 = t1 + ting;
  /* Grazing eclipse:        */
  if ((t1+ting) > midpt)
    t2 = midpt;

  t4 = midpt + width/2;
  t3 = t4 - tegr;
  if ((t4-tegr) < midpt)
    t3 = midpt;

  /* Rp/Rs at ingress and egress: */
  pi = sign(idepth) * sqrt(fabs(idepth));  /* Not to confuse with pi */
  pe = sign(edepth) * sqrt(fabs(edepth));

  for(i=0; i<dims[0]; i++){
    INDd(eclipse,i) = 1.0;
    /* Before ingress: */
    if (INDd(t,i) < t1){
      INDd(eclipse,i) += idepth;
    }
    /* During ingress:                   */
    /* Eq. (1) of Mandel & Agol (2002):  */
    else if (INDd(t,i) < t2  &&  pi != 0){
      z = -2*pi*(INDd(t,i)-t1)/ting + 1 + pi;
      INDd(eclipse,i) += idepth - sign(idepth)/M_PI * (pi*pi*k0(pi,z) + k1(pi,z)
                                   - sqrt(z*z - 0.25*pow(1+z*z-pi*pi,2)));
    }
    /* Totality:       */
    else if (INDd(t,i) < t3){
    }
    /* During egress:  */
    else if (INDd(t,i) < t4  &&  pe != 0){
      z =  2*pe*(INDd(t,i)-t3)/tegr + 1 - pe;
      INDd(eclipse,i) += edepth - sign(edepth)/M_PI * (pe*pe*k0(pe,z) + k1(pe,z)
                                   - sqrt(z*z - 0.25*pow(1+z*z-pe*pe,2)));
    }
    /* After egress:   */
    else if (INDd(t,i) >= t4){
      INDd(eclipse,i) += edepth;
    }
    INDd(eclipse,i) *= flux;
  }
  return Py_BuildValue("N", eclipse);
}


PyDoc_STRVAR(eclipse_lin__doc__,
"Secondary-eclipse light-curve model with linear baseline and \n\
independent ingress and egress depths and slopes.");

static PyObject *eclipse_lin(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, idepth, edepth, ting, tegr, flux, islope, eslope;
  double t1, t2, t3, t4, pi, pe, z;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  midpt  = INDd(params, 0);
  width  = INDd(params, 1);
  idepth = INDd(params, 2);  // Ingress depth
  edepth = INDd(params, 3);  // Egress depth
  ting   = INDd(params, 4);
  tegr   = INDd(params, 5);
  flux   = INDd(params, 6);
  islope = INDd(params, 7);
  eslope = INDd(params, 8);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  /* Time of contact points: */
  t1 = midpt - width/2;
  t2 = t1 + ting;
  /* Grazing eclipse:        */
  if ((t1+ting) > midpt)
    t2 = midpt;

  t4 = midpt + width/2;
  t3 = t4 - tegr;
  if ((t4-tegr) < midpt)
    t3 = midpt;

  /* Rp/Rs at ingress and egress: */
  pi = sign(idepth) * sqrt(fabs(idepth));  /* Not to confuse with pi */
  pe = sign(edepth) * sqrt(fabs(edepth));

  for(i=0; i<dims[0]; i++){
    INDd(eclipse,i) = 1.0;
    /* Before ingress: */
    if (INDd(t,i) < t1){
      INDd(eclipse,i) += idepth + islope*(INDd(t,i)-t1);
    }
    /* During ingress:                   */
    /* Eq. (1) of Mandel & Agol (2002):  */
    else if (INDd(t,i) < t2  &&  pi != 0.0){
      z = -2*pi*(INDd(t,i)-t1)/ting + 1 + pi;
      INDd(eclipse,i) += idepth - sign(idepth)/M_PI * (pi*pi*k0(pi,z) + k1(pi,z)
                                   - sqrt(z*z - 0.25*pow(1+z*z-pi*pi,2)));
    }
    /* Totality:       */
    else if (INDd(t,i) < t3){
    }
    /* During egress:  */
    else if (INDd(t,i) < t4  &&  pe != 0.0){
      z =  2*pe*(INDd(t,i)-t3)/tegr + 1 - pe;
      INDd(eclipse,i) += edepth - sign(edepth)/M_PI * (pe*pe*k0(pe,z) + k1(pe,z)
                                   - sqrt(z*z - 0.25*pow(1+z*z-pe*pe,2)));
    }
    /* After egress:   */
    else if (INDd(t,i) >= t4){
      INDd(eclipse,i) += edepth + eslope*(INDd(t,i)-t4);
    }
    /* Flux normalization: */
    INDd(eclipse,i) *= flux;
  }
  return Py_BuildValue("N", eclipse);
}



PyDoc_STRVAR(eclipse_quad__doc__,
"Secondary-eclipse light-curve model with independent ingress and    \n\
egress depths and quadratic baseline.                                \n\
                                                                     \n\
Parameters                                                           \n\
----------                                                           \n\
params: 1D float ndarray                                             \n\
   Eclipse model parameters:                                         \n\
   - midpt:  Mid-eclipse epoch.                                      \n\
   - width:  Eclipse duration between 1st and 4th contacts (T14).    \n\
   - idepth: Normalized ingress eclipse depth.                       \n\
   - edepth: Normalized egress eclipse depth.                        \n\
   - ting:   Eclipse ingress duration (between 1st and 2nd contacts).\n\
   - tegr:   Eclipse egress duration (between 3rd and 4th contacts). \n\
   - flux:   Stellar flux level (i.e., flux during eclipse).         \n\
   - slope:  Out-of-eclipse flux linear slope.                       \n\
   - quad:   Out-of-eclipse flux quadratic term.                     \n\
t: 1D float ndarray                                                  \n\
   The lightcurve's phase/time points.                               \n\
                                                                     \n\
Returns                                                              \n\
-------                                                              \n\
eclipse: 1D float ndarray                                            \n\
   Mandel & Agol eclipse model evaluated at points t.");


static PyObject *eclipse_quad(PyObject *self, PyObject *args){
  PyArrayObject *t, *eclipse, *params;
  double midpt, width, idepth, edepth, ting, tegr, flux, slope, quad;
  double t1, t2, t3, t4, pi, pe, z;
  int i;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args, "OO", &params, &t)){
    return NULL;
  }

  /* Unpack params:           */
  midpt  = INDd(params, 0);
  width  = INDd(params, 1);
  idepth = INDd(params, 2);
  edepth = INDd(params, 3);
  ting   = INDd(params, 4);
  tegr   = INDd(params, 5);
  flux   = INDd(params, 6);
  slope  = INDd(params, 7);
  quad   = INDd(params, 8);

  dims[0] = (int)PyArray_DIM(t, 0);
  eclipse = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  /* Time of contact points:  */
  t1 = midpt - width/2;
  t2 = t1 + ting;
  /* Grazing eclipse:         */
  if ((t1+ting) > midpt)
    t2 = midpt;

  t4 = midpt + width/2;
  t3 = t4 - tegr;
  if ((t4-tegr) < midpt)
    t3 = midpt;

  /* Rp/Rs at ingress and egress:                                    */
  pi = sign(idepth) * sqrt(fabs(idepth));  /* Not to confuse with pi */
  pe = sign(edepth) * sqrt(fabs(edepth));

  for (i=0; i<dims[0]; i++){
    INDd(eclipse,i) = 1.0;
    /* Before ingress:                   */
    if (INDd(t,i) < t1){
      INDd(eclipse,i) += quad * (INDd(t,i) + t1 - 2*midpt) * (INDd(t,i) - t1)
                         + slope*(INDd(t,i)-t1) + idepth;
    }
    /* During ingress:                   */
    /* Eq. (1) of Mandel & Agol (2002):  */
    else if (INDd(t,i) < t2  &&  pi != 0.0){
      z  = -2*pi*(INDd(t,i)-t1)/ting + 1 + pi;
      INDd(eclipse,i) += idepth - sign(idepth)/M_PI * (pi*pi*k0(pi,z) + k1(pi,z)
                                    - sqrt(z*z - 0.25*pow(1+z*z-pi*pi,2)));
    }
    /* Totality (t2 < t < t3):           */
    else if (INDd(t,i) < t3){  /* Already set INDd(eclipse,i) = 1.0 */
    }
    /* During egress:                    */
    else if (INDd(t,i) < t4  &&  pe != 0.0){
      z  = 2*pe*(INDd(t,i)-t3)/tegr + 1 - pe;
      INDd(eclipse,i) += edepth - sign(edepth)/M_PI * (pe*pe*k0(pe,z) + k1(pe,z)
                                    - sqrt(z*z - 0.25*pow(1+z*z-pe*pe,2)));
    }
    /* After egress:                     */
    else if (INDd(t,i) >= t4){
      INDd(eclipse,i) += quad * (INDd(t,i) + t4 - 2*midpt) * (INDd(t,i) - t4)
                         + slope*(INDd(t,i)-t4) + edepth;
    }
    /* Flux normalization:               */
    INDd(eclipse,i) *= flux;
  }
  return Py_BuildValue("N", eclipse);
}


PyDoc_STRVAR(eclipse__doc__,
             "Eclipse light-curve models for the times of the Webb.\n");


static PyMethodDef eclipse_methods[] = {
  {"mandelecl",    mandelecl,    METH_VARARGS, mandelecl__doc__},
  {"eclipse_flat", eclipse_flat, METH_VARARGS, eclipse_flat__doc__},
  {"eclipse_lin",  eclipse_lin,  METH_VARARGS, eclipse_lin__doc__},
  {"eclipse_quad", eclipse_quad, METH_VARARGS, eclipse_quad__doc__},
  {NULL,           NULL,         0,            NULL}
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "_eclipse", eclipse__doc__, -1, eclipse_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__eclipse (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_eclipse(void){
  Py_InitModule3("_eclipse", eclipse_methods, eclipse__doc__);
  import_array();
}
#endif
