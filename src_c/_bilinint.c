// Copyright (c) 2021 Patricio Cubillos
// puppies is open-source software under the MIT license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include "ind.h"


PyDoc_STRVAR(bilinint__doc__,
"This function fits the intra-pixel sensitivity effect using bilinear\n\
interpolation to fit mean binned flux vs position.                   \n\
                                                                     \n\
Parameters                                                           \n\
----------                                                           \n\
flux:  1D float ndarray                                              \n\
   Flux of each data point.                                          \n\
model:  1D float ndarray                                             \n\
   Model of all other data systematics.                              \n\
knotpts:  1D integer ndarray                                         \n\
   Array with the datapoint indices sorted per knot.                 \n\
knotpsize:  1D integer ndarray                                       \n\
   Number of data points for each knot.                              \n\
kploc:  1D integer ndarray                                           \n\
   Index in knotpts of the first data-point index of each knot.      \n\
binloc:  1D integer ndarray                                          \n\
   Indices of the knot to the lower left of the datapoints.          \n\
ydist:  1D float ndarray                                             \n\
   Normalized Y-axis distance to the binloc knot.                    \n\
xdist:  1D float ndarray                                             \n\
   Normalized X-axis distance to the binloc knot.                    \n\
xsize:  Integer                                                      \n\
   Number of knots along the x axis.                                 \n\
retmap:  Boolean                                                     \n\
   If True return the BLISS map.                                     \n\
retbinstd:  Boolean                                                  \n\
   If True, return the standard deviation of the BLISS map.          \n\
                                                                     \n\
Returns                                                              \n\
-------                                                              \n\
ipflux:  1D float ndarray                                            \n\
   Normalized intra-pixel flux variation for each datapoint.         \n\
blissmap:  1D float ndarray                                          \n\
   The BLISS map.                                                    \n\
binstd:  1d float ndarray                                            \n\
   BLISS map standard deviation.");


static PyObject *bilinint(PyObject *self, PyObject *args){
    PyArrayObject *flux, *model, *knotpts, *ydist, *xdist,
                  *knotsize, *kploc, *binloc;
    npy_intp nknots[1], npts[1];

    int ibl, ibr, itl, itr;  /* Indices of knots around a data point          */
    int i, j, xsize, arsize, iknot, idata, counter;
    double mean, std, meanbinflux;

    /* Return flags:                                                  */
    PyObject *retmap=Py_False,
             *retbinstd=Py_False;
    /* Returned arrays:                                               */
    PyArrayObject *ipflux, *blissmap, *binstd;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOi|OO",
            &flux, &model, &knotpts,
            &knotsize, &kploc, &binloc, &ydist, &xdist, &xsize,
            &retmap, &retbinstd))
        return NULL;

    nknots[0] = (int)PyArray_DIM(kploc, 0);  /* Number of knots       */
    npts  [0] = (int)PyArray_DIM(flux,  0);  /* Number of data points */

    /* Allocate output intra-pixel flux array:                        */
    ipflux   = (PyArrayObject *) PyArray_SimpleNew(1, npts, NPY_DOUBLE);
    /* Allocate output BLISS map and BLISS map std:                   */
    blissmap = (PyArrayObject *) PyArray_SimpleNew(1, nknots, NPY_DOUBLE);
    binstd   = (PyArrayObject *) PyArray_SimpleNew(1, nknots, NPY_DOUBLE);

    counter = 0;      /* Number of used BLISS knots in calculation    */
    meanbinflux = 0;  /* BLISS overall mean flux                      */

    /* Calculate the mean flux for each BLISS knot:                   */
    for(i=0; i < (int)nknots[0]; i++){
        if(INDi(knotsize, i) > 0){
            arsize = INDi(knotsize, i);  /* Data points in knot           */
            iknot  = INDi(kploc,    i);  /* Index in knotpts              */
            mean = 0;
            /* Calculate the mean in the knot:                            */
            for(j=iknot; j<iknot+arsize; j++){
                idata = INDi(knotpts, j);  /* Data-point index              */
                mean += (INDd(flux, idata)/INDd(model, idata));
            }
            mean /= (double)arsize;
            INDd(blissmap, i) = mean;

            /* Calculate the standard deviation if requested:             */
            if(PyObject_IsTrue(retbinstd) == 1){
                std = 0;
                for(j=iknot; j<iknot+arsize; j++){
                    idata = INDi(knotpts, j);
                    std +=
                        pow(((INDd(flux,idata) / INDd(model,idata)) - mean), 2);
                }
                INDd(binstd, i) = sqrt(std / (double)arsize);
            }
            meanbinflux += mean;
            counter += 1;
        }
        else{
            INDd(blissmap, i) = 0;
            INDd(binstd, i) = 0;
        }
    }
    /* BLISS overall mean flux:                                               */
    meanbinflux /= (double) counter;

    /* Normalize the mean flux:                                               */
    for(i=0; i<nknots[0]; i++){
        INDd(blissmap, i) /= meanbinflux;
        INDd(binstd, i) /= meanbinflux;
    }

    /* Compute the BLISS map correction:                                      */
    for(i=0; i<npts[0]; i++){
        ibl = INDi(binloc, i);  /* Bottom-left knot */
        ibr = ibl + 1;          /* Bottom-right knot */
        itl = ibl + xsize;      /* Top-left knot */
        itr = itl + 1;          /* Top-right knot */
        /* Bi-linear interpolation: */
        INDd(ipflux, i) =
            INDd(blissmap, ibl) * (1.0-INDd(ydist,i)) * (1.0-INDd(xdist,i)) +
            INDd(blissmap, ibr) * (1.0-INDd(ydist,i)) *      INDd(xdist,i)  +
            INDd(blissmap, itl) *      INDd(ydist,i)  * (1.0-INDd(xdist,i)) +
            INDd(blissmap, itr) *      INDd(ydist,i)  *      INDd(xdist,i);
    }

    /* Return BLISS flux, map, and std arrays as requested:                   */
    if(PyObject_IsTrue(retmap) == 0 && PyObject_IsTrue(retbinstd) == 0){
        Py_XDECREF(blissmap);
        Py_XDECREF(binstd);
        return PyArray_Return(ipflux);
    }
    else if (PyObject_IsTrue(retmap) == 1 && PyObject_IsTrue(retbinstd)==1){
        return Py_BuildValue("NNN", ipflux, blissmap, binstd);
    }
    else if (PyObject_IsTrue(retmap) == 1){
        Py_XDECREF(binstd);
        return Py_BuildValue("NN", ipflux, blissmap);
    }
    else{
        Py_XDECREF(blissmap);
        return Py_BuildValue("NN", ipflux, binstd);
    }
}


/* Module's doc string */
PyDoc_STRVAR(
    bilinint_mod__doc__,
    "Bi-linearly interpolated subpixel sensitivity map.\n");


static PyMethodDef bilinint_methods[] = {
    {"bilinint", bilinint, METH_VARARGS, bilinint__doc__},
    {NULL, NULL, 0, NULL}
};


/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_bilinint",
    bilinint_mod__doc__,
    -1,
    bilinint_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__bilinint (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
