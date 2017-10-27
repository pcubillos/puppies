void cresize(PyArrayObject *idata, PyArrayObject *data,
             int expand, int ymin, int xmin, int my, int mx){
  // resize a 2D data image using bi-linear interpolation.
  // store result in idata.
  int i, j, n, m;
  double x[mx], y[my], xx[mx], yy[my];
  double t, u;

  if (expand==1){ /* No resize */
    for (j=0; j<my; j++)
      for (i=0; i<mx; i++)
        IND2d(idata,j,i) = IND2d(data,(j+ymin),(i+xmin));
    return;
  }

  for (j=0; j<my; j++)  y[j]  = ymin + j/expand;
  for (i=0; i<mx; i++)  x[i]  = xmin + i/expand;
  //for (j=0; j<my; j++)  y[j]  = ymin + floor(j*1.0/expand);
  //for (i=0; i<mx; i++)  x[i]  = xmin + floor(i*1.0/expand);
  for (j=0; j<my; j++)  yy[j] = fmod(j*1.0/expand, 1);
  for (i=0; i<mx; i++)  xx[i] = fmod(i*1.0/expand, 1);

  for (m=0; m<my; m++){
    u = yy[m];
    j = y[m];
    if (m == my-1){
      j-=1;
      u=1.0;
    }
    for (n=0; n<mx; n++){
      t = xx[n];
      i = x[n];
      if (n==mx-1){
        i-=1;
        t=1.0;
      }
      // interpolated value
      IND2d(idata,m,n) =  (1.0-t)*(1.0-u)*IND2d(data,j,  i  )
                        +      t *     u *IND2d(data,(j+1),(i+1))
                        + (1.0-t)*     u *IND2d(data,(j+1),i)
                        +      t *(1.0-u)*IND2d(data,j,  (i+1));
    }
  }
  return;
}


void resize_mask(PyArrayObject *idata, PyArrayObject *data,
                 int expand, int ymin, int xmin, int my, int mx){
  /* Resize for a mask array (char type), return 0 if value < 1.0 */
  int i, j, m, n;
  int x[mx], y[my], xx[mx], yy[my];

  if (expand==1){ /* No resize */
    for (j=0; j<my; j++)
      for (i=0; i<mx; i++)
        IND2b(idata,j,i) = IND2b(data,(j+ymin),(i+xmin));
    return;
  }

  for (j=0; j<my; j++)  y[j]  = ymin + j/expand;
  for (i=0; i<mx; i++)  x[i]  = xmin + i/expand;
  for (j=0; j<my; j++)  yy[j] = (int)(j%expand!=0);
  for (i=0; i<mx; i++)  xx[i] = (int)(i%expand!=0);

  for (m=0; m<my; m++){
    j = y[m];
    for (n=0; n<mx; n++){
      i = x[n];
      IND2b(idata,m,n) = IND2b(data,j,i);
      if (yy[m])
        IND2b(idata,m,n) &= IND2b(data,(j+1),i);
      if (xx[n])
        IND2b(idata,m,n) &= IND2b(data,j,(i+1));
      if (yy[m] && xx[n])
        IND2b(idata,m,n) &= IND2b(data,(j+1),(i+1));
    }
  }
  return;
}

void resize_art(PyArrayObject *idata, PyArrayObject *data,
                 int expand, int ymin, int xmin, int my, int mx){
  /* Resize for a mask array (char type), return 0 if value < 1.0 */
  int i, j, m, n;
  int x[mx], y[my], xx[mx], yy[my];

  if (expand==1){ /* No resize */
    for (j=0; j<my; j++)
      for (i=0; i<mx; i++)
        IND2b(idata,j,i) = IND2b(data,(j+ymin),(i+xmin));
    return;
  }

  for (j=0; j<my; j++)  y[j]  = ymin + j/expand; //floor(j*1.0/expand);
  for (i=0; i<mx; i++)  x[i]  = xmin + i/expand; //floor(i*1.0/expand);
  for (j=0; j<my; j++)  yy[j] = (int)(j%expand==0);
  for (i=0; i<mx; i++)  xx[i] = (int)(i%expand==0);

  for (m=0; m<my; m++){
    j = y[m];
    for (n=0; n<mx; n++){
      i = x[n];
      IND2b(idata,m,n) &= IND2b(data,j,i);
      if (yy[m])
        IND2b(idata,m,n) &= IND2b(data,(j+1),i);
      if (xx[n])
        IND2b(idata,m,n) &= IND2b(data,j,(i+1));
      if (yy[m] && xx[n])
        IND2b(idata,m,n) &= IND2b(data,(j+1),(i+1));
    }
  }
  return;
}
