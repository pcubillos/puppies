/*
Statistical functions:
 median:  calcualte median of an array
 meanerr: calculate weighted mean of an array, and its uncertainty

Sub-routines:
 myselect: get the k-th largest value in an array

Modification history:
2012-04-24  patricio  First implementation
*/

double myselect(double *arr, int k, int n){
  // get the k-th largest value in the first n elements of array
  // side effects: modifies arr
  // from Numerical Recipes
  int i, ir, j, l, mid;
  double a;
  l  = 0;   // first element index
  ir = n-1; // last element  index
  for (;;) {
    if (ir <= l+1) {  // 1 or 2 elements left to compare:
      if (ir == l+1 && arr[ir] < arr[l]) // 2 unsorted elements
        SWAP(&arr[l], &arr[ir]);
      return arr[k];  // and we are done!
    }
    else {
      mid = (l+ir)/2;           // midpoint between l and ir
      SWAP(&arr[mid], &arr[l+1]);  
      if (arr[l] > arr[ir])        // make arr[l  ] < arr[ ir]
        SWAP(&arr[l], &arr[ir]);
      if (arr[l+1] > arr[ir])      // make arr[l+1] < arr[ ir]
        SWAP(&arr[l+1], &arr[ir]);
      if (arr[l] > arr[l+1])       // make arr[l  ] < arr[l+1]
        SWAP(&arr[l], &arr[l+1]);
      i = l+1;    // now partition the array:
      j = ir;
      a = arr[l+1]; // the partitioning element
      for (;;) {
        do i++; while (arr[i] < a && i != n-1);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(&arr[i], &arr[j]);
      }
      arr[l+1]= arr[j];
      arr[j]  = a;
      if (j >= k) ir= j-1;
      if (j <= k) l = i;
    }
  }
}


double median(double *arr, int n){
  // find the median of the n first elements of arr
  double median;
  median =  myselect(arr, n/2, n);
  if (fmod(n,2) == 0)  // if we have even number of elements
    median = (myselect(arr, n/2-1, n/2) + median)/2.0;
   return median;
}


double meanerr(double *data, double *derr, int n, double *err, int *status){
  // calculate the weighted mean and uncertainty of the first n elements of data
  double weight;
  int i, nnan=0, nzero=0;
  double mean=0.0, sum_weights=0.0;
  
  for (i=0; i<n; i++){
    if (isnan(data[i]) || isnan(derr[i]) || isinf(data[i]) || isinf(derr[i]))
      nnan++;          // corrupted data?
    else{
      if (derr[i] <= 0.000001) // is error zero? FINDME: <= 0.0 doesn't work!
	nzero++;
      else{            // everything is fine
	weight = 1.0/pow(derr[i], 2.0);  //  weight = 1 / err^2
	mean += data[i]*weight;
	sum_weights += weight;
      }
    }
  }

  // the mean
  mean = mean/sum_weights;

  // the error in the mean
  *err = sqrt(1./sum_weights);

  // the status
  *status = 0;    // status ok, unless ...
  if (nnan  == n)  // all NaNs
    *status |= 1;
  if (nzero == n)  // all errors == 0.0
    *status |= 2;
  if ((nnan + nzero) == n) // all data is in someway bad
    *status |= 4;

  return mean;
}
