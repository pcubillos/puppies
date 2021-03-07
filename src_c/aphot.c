#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>

#include "ind.h"
#include "utils_c.h"
#include "stats_c.h"
#include "cdisk.h"
#include "cresize.h"

PyDoc_STRVAR(aphot__doc__,
"Circular-interpolated aperture photometry.                             \n\
                                                                        \n\
Parameters                                                              \n\
----------                                                              \n\
image: 2D float ndarray                                                 \n\
  Input image.                                                          \n\
uncert: 2D float ndarray                                                \n\
  Uncerainties of the input image.                                      \n\
gpmask: 2D bool ndarray                                                 \n\
  Good pixel mask of the input image.                                   \n\
yctr: Float                                                             \n\
  Center of the aperture along the first dimension (in pixels).         \n\
  The origin is at the center of the first pixel.                       \n\
xctr: Float                                                             \n\
  Center of the aperture along the second dimension (in pixels).        \n\
  The origin is at the center of the first pixel.                       \n\
photap: Float                                                           \n\
  Aperture radius (in pixels).                                          \n\
skyin: Float                                                            \n\
  Inner sky annulus radius (in pixels).                                 \n\
skyout: Float                                                           \n\
  Outer sky annulus radius (in pixels).                                 \n\
skyfrac: Float                                                          \n\
  Minimum acceptable fraction of good sky pixels.                       \n\
expand: Integer                                                         \n\
  Resampling interpolation factor.                                      \n\
med: Bool                                                               \n\
  If True, compute the sky level as the median instead of the mean.     \n\
                                                                        \n\
Return                                                                  \n\
------                                                                  \n\
aplev: Float                                                            \n\
   Total sky-subtracted flux inside the aperture.                       \n\
aperr: Float                                                            \n\
   Uncertainty of aplev.                                                \n\
napix: Float                                                            \n\
   Number of pixels in the aperture.                                    \n\
skylev: Float                                                           \n\
   Mean or median flux in the sky annulus (see med parameter).          \n\
skyerr: Float                                                           \n\
   Uncertainty of skylev.                                               \n\
nsky: Integer                                                           \n\
   Number of good sky pixels.                                           \n\
nskyideal: Integer                                                      \n\
   Ideal number of sky pixels.                                          \n\
status: Integer                                                         \n\
   Status binary flag:                                                  \n\
   - If status ==  0:  Good result, no warnings.                        \n\
   - If status &=  1:  NaNs or Inf pixels in aperture                   \n\
   - If status &=  2:  No good pixels in aperture                       \n\
   - If status &=  4:  Masked pixels in aperture                        \n\
   - If status &=  8:  Out of bounds aperture                           \n\
   - If status &= 16:  Sky fraction condition unfulfilled               \n\
   - If status &= 32:  No good pixels in sky                            \n\
                                                                        \n\
Examples                                                                \n\
--------                                                                \n\
>>> import sys                                                          \n\
>>> import numpy as np                                                  \n\
>>> import matplotlib.pyplot as plt                                     \n\
                                                                        \n\
>>> sys.path.append('puppies/lib/')                                     \n\
>>> import aphot as ap                                                  \n\
>>> import gauss as g                                                   \n\
                                                                        \n\
>>> test      = 0                                                       \n\
>>> ntest     = 11                                                      \n\
>>> testout   = np.zeros((5, ntest))                                    \n\
>>> testright = np.zeros((5, ntest))                                    \n\
>>> ny, nx    = 50, 50                                                  \n\
>>> sig       = 3.0, 3.0                                                \n\
>>> ctr       = 25.8, 25.2                                              \n\
>>> photap    = 12                                                      \n\
>>> skyin     = 12                                                      \n\
>>> skyout    = 15                                                      \n\
>>> ampl      = 1000.0                                                  \n\
>>> sky       = 100.0                                                   \n\
>>> h = ampl/(2*np.pi*sig[0]*sig[1])                                    \n\
                                                                        \n\
>>> image = g.gauss2D(ny, nx, ctr[0], ctr[1], sig[0], sig[1], h, sky)   \n\
>>> plt.figure(1)                                                       \n\
>>> plt.clf()                                                           \n\
>>> plt.pcolor(image, cmap=plt.cm.gray)                                 \n\
>>> plt.xlabel('X coordinate')                                          \n\
>>> plt.ylabel('Y coordinate')                                          \n\
>>> plt.axis([0, nx-1, 0, ny-1])                                        \n\
>>> plt.colorbar()                                                      \n\
                                                                        \n\
>>> skyfrac = 0.0                                                       \n\
>>> expand = 1                                                          \n\
>>> med = False                                                         \n\
>>> mask   = np.ones((ny,nx), bool)                                     \n\
>>> uncert = np.ones((ny,nx), double)                                   \n\
                                                                        \n\
>>> image = np.tile(sky, (ny,nx))                                       \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, uncert, mask, ctr[0], ctr[1],               \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]         \n\
>>> test += 1                                                           \n\
>>> # Flat image, no stellar flux, only sky                             \n\
                                                                        \n\
>>> image = g.gauss2D(ny, nx, ctr[0], ctr[1], sig[0], sig[1], h, sky)   \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, uncert, mask, ctr[0], ctr[1],               \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]         \n\
>>> test += 1                                                           \n\
>>> # A little flux leaks from aperture to sky, the rest is right.      \n\
                                                                        \n\
>>> mask = np.ones((ny,nx), bool)                                       \n\
>>> mask[24,24] = False                                                 \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, uncert, mask, ctr[0], ctr[1],               \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, 0, skylev, 0, status]                  \n\
>>> test += 1                                                           \n\
>>> # We use the bad value since it's in the aperture, but we flag it.  \n\
                                                                        \n\
>>> image[25,24] = np.nan                                               \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, uncert, mask, ctr[0], ctr[1],               \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, 0, skylev, 0, status]                  \n\
>>> test += 1                                                           \n\
>>> # We can't use a NaN! Flagged, and value changes.                   \n\
>>> # Bad value still flagged.                                          \n\
                                                                        \n\
>>> ctr2 = [48.8, 48.2]                                                 \n\
>>> image2 = g.gauss2D(ny, nx, ctr2[0], ctr2[1], sig[0], sig[1], h, sky)\n\
                                                                        \n\
>>> plt.figure(2)                                                       \n\
>>> plt.clf()                                                           \n\
>>> plt.title('Gaussian')                                               \n\
>>> plt.xlabel('X coordinate')                                          \n\
>>> plt.ylabel('Y coordinate')                                          \n\
>>> plt.pcolor(image2, cmap=plt.cm.gray)                                \n\
>>> plt.axis([0, nx-1, 0, ny-1])                                        \n\
>>> plt.colorbar()                                                      \n\
>>> plt.show()                                                          \n\
                                                                        \n\
>>> mask = np.ones((ny,nx), bool)                                       \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image2, uncert, mask, ctr2[0], ctr2[1],            \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, 0, skylev, 0, status]                  \n\
>>> test += 1                                                           \n\
>>> # Flagged that we're off the image.                                 \n\
                                                                        \n\
>>> skyfrac = 0.5                                                       \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image2, uncert, mask, ctr2[0], ctr2[1],            \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, 0, skylev, 0, status]                  \n\
>>> test += 1                                                           \n\
>>> # Flagged that we are off the image and have insufficient sky.      \n\
>>> # Same numbers.                                                     \n\
                                                                        \n\
>>> image = g.gauss2D(ny, nx, ctr[0], ctr[1], sig[0], sig[1], h, sky)   \n\
>>> imerr = np.sqrt(image)                                              \n\
>>> mask = np.ones((ny,nx), bool)                                       \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, imerr, mask, ctr[0], ctr[1],                \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]         \n\
>>> test += 1                                                           \n\
>>> # Estimates for errors above.  Basic numbers don't change.          \n\
                                                                        \n\
>>> imerr[25, 38] = 0                                                   \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, imerr, mask, ctr[0], ctr[1],                \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]         \n\
>>> test += 1                                                           \n\
>>> # The zero-error pixel is ignored in the sky average.               \n\
>>> # Small changes result.                                             \n\
                                                                        \n\
>>> imerr[25, 38] = np.nan                                              \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, imerr, mask, ctr[0], ctr[1],                \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]         \n\
>>> test += 1                                                           \n\
>>> # The NaN in the sky error is ignored, with the same result.        \n\
                                                                        \n\
>>> image[25, 38] = np.nan                                              \n\
>>> imerr = sqrt(image)                                                 \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, imerr, mask, ctr[0], ctr[1],                \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]         \n\
>>> test += 1                                                           \n\
>>> # The NaN in the sky data is ignored, with the same result.         \n\
                                                                        \n\
>>> image = g.gauss2D(ny, nx, ctr[0], ctr[1], sig[0], sig[1], h, sky)   \n\
>>> imerr  = sqrt(image)                                                \n\
>>> mask = np.ones((ny,nx), bool)                                       \n\
>>> expand = 5                                                          \n\
>>> aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status = \\\n\
            ap.aphot(image, imerr, mask, ctr[0], ctr[1],                \n\
                     photap, skyin, skyout, skyfrac, expand, med)       \n\
>>> print('{:8.3f} {:8.4f} {:7.2f} {:9.5f} {:9.5f} {:6.2f} {:6.2f} {:d}'\n\
     .format(aplev, aperr, napix, skylev, skyerr, nsky, nskyid, status))\n\
>>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]         \n\
>>> test += 1                                                           \n\
>>> # Slight changes.                                                   \n\
                                                                        \n\
>>> skyerrest = np.sqrt(sky/(np.pi * (skyout**2 - skyin**2)))           \n\
>>> #    0.62687732                                                     \n\
>>> aperrest = np.sqrt(ampl + np.pi * photap**2 * (sky+skyerrest**2))   \n\
>>> # Note that background flux of 100 contributes a lot!               \n\
>>> #  215.44538                                                        \n\
                                                                        \n\
>>> print('\\nCorrect:' )                                               \n\
>>> print('{:8.3f}  {:7.3f}  {:12.8f} {:12.8f} {:2d}'.                  \n\
             format(ampl, aperrest, sky, skyerrest, 0))                 \n\
>>> print( '\\nTest results:')                                          \n\
>>> for i in np.arange(testout.shape[1]):                               \n\
>>>   print('{:8.3f}  {:7.3f}  {:12.8f} {:12.8f} {:2.0f}'.              \n\
            format(*list(testout[:,i])))");


/* The wrapper to the underlying C function */
static PyObject *aphot(PyObject *self, PyObject *args){
  PyArrayObject *image,  *uncert, *gpmask,               /* Inputs   */
                *idata, *iunc, *imask,                   /* Expanded */
                *indisk, *outdisk, *stardisk, *idealim;  /* Disks    */

  double xctr,  yctr,  photap,  skyin,  skyout, skyfrac, /* Inputs   */
         ixctr, iyctr, iphotap, iskyin, iskyout,         /* Expanded */
         aplev=0.0, aperr=0.0, apvar=0.0,
         napix, skylev, skylevunc, nsky, nskyideal;      /* Outputs  */

  int diskstatus, ndiskin, ndiskout, ndiskap,            /* Utils    */
      i, j, nx, ny, mx, my, expand, med, isis, dummy,
      xmin, xmax, ymin, ymax,
      naper=0, nskypix=0, nmask=0, nbad=0, nskyipix, status=0;

  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "OOOddddddii", &image, &uncert, &gpmask,
        &yctr, &xctr, &photap, &skyin, &skyout, &skyfrac, &expand, &med))
    return NULL;

  /* Dimensions of the input 2D array: */
  ny = (int)PyArray_DIM(image, 0);
  nx = (int)PyArray_DIM(image, 1);

  /* Work only around the target:      */
  ymin = MAX(0,  (int)(round(yctr)-skyout-2));
  xmin = MAX(0,  (int)(round(xctr)-skyout-2));
  ymax = MIN(ny, (int)(round(yctr)+skyout+2));
  xmax = MIN(nx, (int)(round(xctr)+skyout+2));
  ny = ymax - ymin;
  nx = xmax - xmin;

  /* Interpolation:   */
  my = dims[0] = ny + (ny-1)*(expand-1);
  mx = dims[1] = nx + (nx-1)*(expand-1);
  idata = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  iunc  = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  imask = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);
  cresize(     idata, image,  expand, ymin, xmin, my, mx);
  cresize(     iunc,  uncert, expand, ymin, xmin, my, mx);
  resize_mask(imask, gpmask, expand,  ymin, xmin, my, mx);

  /* Expand lengths:  */
  iphotap = photap      * expand;
  iskyin  = skyin       * expand;
  iskyout = skyout      * expand;
  iyctr   = (yctr-ymin) * expand;
  ixctr   = (xctr-xmin) * expand;

  /* Make disks:      */
  indisk   = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);
  outdisk  = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);
  stardisk = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);
  cdisk(indisk,   iskyin,  iyctr, ixctr, my, mx, &diskstatus, &ndiskin);
  cdisk(outdisk,  iskyout, iyctr, ixctr, my, mx, &diskstatus, &ndiskout);
  cdisk(stardisk, iphotap, iyctr, ixctr, my, mx, &diskstatus, &ndiskap);

  /* Allocate useful data in 1D arrays: */
  double skydata[ndiskout-ndiskin];
  double  skyerr[ndiskout-ndiskin];

  for   (j=0; j<my; j++)
    for (i=0; i<mx; i++){
      /* Pixels in annulus:             */
      if (IND2b(outdisk,j,i) && !IND2b(indisk,j,i) ){
        if (!isnan(IND2d(idata,j,i)) &&
            !isinf(IND2d(idata,j,i)) &&
            IND2b(imask,j,i)){
          nskypix++;
          skydata[nskypix] = IND2d(idata,j,i);
          skyerr [nskypix] = IND2d(iunc, j,i);
        }
      }
      /* Pixels in aperture:            */
      if (IND2b(stardisk,j,i)){
        if (!IND2b(imask,j,i))
          nmask++;
        if (isnan(IND2d(idata,j,i)) || isinf(IND2d(idata,j,i)))
          nbad++;
        else{
          aplev += IND2d(idata,j,i);
          apvar += IND2d(iunc, j,i)*IND2d(iunc,j,i);
          naper++;
        }
      }
    }
  Py_XDECREF(idata); Py_XDECREF(iunc); Py_XDECREF(imask);
  Py_XDECREF(indisk); Py_XDECREF(outdisk); Py_XDECREF(stardisk);

  /* Re-scaled number of pixels in sky: */
  napix = 1.0*naper  /(expand*expand);
  nsky  = 1.0*nskypix/(expand*expand);

  /* Ideal sky calculation:             */
  isis = (int)(2*ceil(iskyout) + 3);             /* ideal sky image size */
  iyctr = fmod(iyctr,1.0) + ceil(iskyout) + 1.0; /* ideal sky center y   */
  ixctr = fmod(ixctr,1.0) + ceil(iskyout) + 1.0; /* ideal sky center x   */

  dims[0] = dims[1] = isis;
  idealim = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_BOOL);
  /* Number of pixels in sky annulus:   */
  cdisk(idealim, iskyout, iyctr, ixctr, isis, isis, &dummy, &nskyipix);
  nskyideal = (double)nskyipix;
  cdisk(idealim, iskyin,  iyctr, ixctr, isis, isis, &dummy, &nskyipix);
  nskyideal = (nskyideal - nskyipix) / (expand*expand);
  Py_XDECREF(idealim);

  /* Mean/median sky level: */
  skylev = meanerr(&skydata[0], &skyerr[0], nskypix, &skylevunc, &dummy);
  skylevunc *= expand;
  if (med)
    skylev = median(&skydata[0], nskypix);

  /* Photometry:    */
  aplev = (aplev - skylev*naper)/(expand*expand);
  aperr = sqrt(apvar + naper*skylevunc*skylevunc)/expand;

  /* Status report: */
  if (nbad > 0)                    /* NaNs or Inf pixels in aperture     */
    status |= 1;
  if (naper == 0)                  /* No good pixels in aperture         */
    status |= 2;
  if (nmask > 0)                   /* Masked pixels in aperture          */
    status |= 4;
  if (diskstatus)                  /* Out of bounds aperture             */
    status |= 8;
  if (nsky < skyfrac * nskyideal)  /* Sky fraction condition unfulfilled */
    status |= 16;
  if (nskypix == 0)                /* No good pixels in sky              */
    status |= 32;

  return Py_BuildValue("[d,d,d,d,d,d,d,i]", aplev, aperr, napix, skylev,
           skylevunc, nsky, nskyideal, status);
}


/* Module's doc string */
PyDoc_STRVAR(aphotmod__doc__, "Aperture photometry C extension.");

/* A list of all the methods defined by this module.        */
static PyMethodDef aphot_methods[] = {
    {"aphot",  aphot, METH_VARARGS, aphot__doc__},
    {NULL,     NULL,   0,            NULL}      /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "aphot", aphotmod__doc__, -1, aphot_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_aphot (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void initaphot(void){
  Py_InitModule3("aphot", aphot_methods, aphotmod__doc__);
  import_array();
}
#endif
