void disk(char **disk, double radius, double yctr, double xctr,
          int ny, int nx, int *status, int *ndisk){
  int i, j, n=0;

  /* Alert if the center lies outside the image: */
  if ( (yctr-radius) < 0 || (yctr+radius) > (ny-1) ||
       (xctr-radius) < 0 || (xctr+radius) > (nx-1) )
    *status = 1;
  else
    *status = 0;

  for   (i=0; i<ny; i++)
    for (j=0; j<nx; j++){
      /* Is the point disk[i][j] inside the disk? */
      disk[i][j] = (i-yctr)*(i-yctr) + (j-xctr)*(j-xctr) <= radius*radius;
      n += disk[i][j];
    }

  /* Set the number of pixels within radius in ndisk: */
  *ndisk = n;

  return;
}
