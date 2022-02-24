// Copyright (c) 2021 Patricio Cubillos
// puppies is open-source software under the MIT license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include "ind.h"


PyDoc_STRVAR(mandeltr__doc__,
"Compute a primary transit light curve using the non-linear    \n\
limb-darkening equations for a 'small planet' (rprs <= 0.1), as\n\
provided by Mandel & Agol (2002).                              \n\
                                                               \n\
Parameters                                                     \n\
----------                                                     \n\
params: 1D float ndarray                                       \n\
    epoch: Transit mid time.                                   \n\
    rprs:  Planet radius / stellar radius.                     \n\
    cosi:  Cosine of the inclination.                          \n\
    ars:   Semi-major axis / stellar radius.                   \n\
    flux:  Out-of-transit flux.                                \n\
    per:   Period in same units as time.                       \n\
    c1:    Limb-darkening coefficient.                         \n\
    c2:    Limb-darkening coefficient.                         \n\
    c3:    Limb-darkening coefficient.                         \n\
    c4:    Limb-darkening coefficient.                         \n\
t: 1D float ndarray                                            \n\
    Phase/time points where to evaluate the model.             \n\
                                                               \n\
Returns                                                        \n\
-------                                                        \n\
Flux for each point in time.                                   \n\
                                                               \n\
Notes                                                          \n\
-----                                                          \n\
For quadratic limb darkening:                                  \n\
    I(r) = 1 - gamma1*(1-mu) - gamma2*(1-mu)**2                \n\
where gamma1 + gamma2 < 1.  We have:                           \n\
    c1 = c3 = 0,                                               \n\
    c2 = gamma1 + 2*gamma2,                                    \n\
    c4 = -gamma2                                               \n\
                                                               \n\
Developers                                                     \n\
----------                                                     \n\
Kevin Stevenson,   UCF                                         \n\
Nate Lust,         UCF                                         \n\
Patricio Cubillos, IWF");

static PyObject *mandeltr(PyObject *self, PyObject *args){
    PyArrayObject *t, *y, *params;
    double epoch, rprs, cosi, ars, flux, x;
    double per, z, c1, c2, c3, c4, Sigma4, I1star, sig1, sig2, I2star, mod;
    int i;
    npy_intp dims[1];

    if(!PyArg_ParseTuple(args, "OO", &params, &t))
        return NULL;

    epoch = INDd(params,0);
    rprs  = INDd(params,1);
    cosi  = INDd(params,2);
    ars   = INDd(params,3);
    flux  = INDd(params,4);
    per   = INDd(params,5);
    c1    = INDd(params,6);
    c2    = INDd(params,7);
    c3    = INDd(params,8);
    c4    = INDd(params,9);

    dims[0] = (int)PyArray_DIM(t,0);

    y = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    Sigma4 = (1 - c1/5 - c2/3 - 3*c3/7 - c4/2);

    for(i=0; i<dims[0]; i++){
        INDd(y,i) = 1;
        if(rprs != 0){
            mod = (INDd(t,i)-epoch) - floor((INDd(t,i)-epoch)/per)*per;
            if((mod > (per/4)) && (mod < (3*per/4))){
                z = ars;
            }
            else{
                z = ars*sqrt(
                    pow(sin(2*M_PI*(INDd(t,i)-epoch)/per),2)
                    + pow(cosi*cos(2*M_PI*(INDd(t,i)-epoch)/per),2));
            }
            /* Ingress or egress: */
            if(z>(1-rprs) && z<=(1+rprs)){
                x = 1 - pow((z-rprs),2);
                I1star =
                    1.0
                    - c1*(1-4/5.0*sqrt(sqrt(x)))
                    - c2*(1-2/3.0*sqrt(x))
                    - c3*(1-4/7.0*sqrt(sqrt(x*x*x)))
                    - c4*(1-4/8.0*x);
                INDd(y,i) =
                    1.0 - I1star*(rprs*rprs*acos((z-1)/rprs)
                    - (z-1)*sqrt(rprs*rprs-(z-1)*(z-1)))/M_PI/Sigma4;
              }
            /* t2 - t3 (except at z=0): */
            else if(z <= (1-rprs)  &&  z != 0){
                sig1 = sqrt(sqrt(1-pow((z-rprs),2)));
                sig2 = sqrt(sqrt(1-pow((z+rprs),2)));
                I2star = 1 - c1*(1+(pow(sig2,5)-pow(sig1,5))/5.0/rprs/z)
                           - c2*(1+(pow(sig2,6)-pow(sig1,6))/6.0/rprs/z)
                           - c3*(1+(pow(sig2,7)-pow(sig1,7))/7.0/rprs/z)
                           - c4*(rprs*rprs+z*z);
                INDd(y,i) = 1-rprs*rprs*I2star/Sigma4;
              }
            /* z=0 (midpoint): */
            else if(z==0){
                INDd(y,i) = 1 - rprs*rprs/Sigma4;
            }
        }
        INDd(y,i) *= flux;
    }
    return Py_BuildValue("N", y);
}


PyDoc_STRVAR(
    mandeltr_mod__doc__,
    "Mandel-Agol limb-darkening transit light-curve model.\n");


static PyMethodDef mandeltr_methods[] = {
    {"mandeltr", mandeltr, METH_VARARGS, mandeltr__doc__},
    {NULL, NULL, 0, NULL}
};


/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_mandeltr",
    mandeltr_mod__doc__,
    -1,
    mandeltr_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__mandeltr (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
