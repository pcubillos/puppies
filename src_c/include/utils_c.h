/*
Definitions:
 define bool type

Utility functions:
 MAX : return largest  value between two inputs
 MIN : return smallest value between two inputs
 SWAP: swap two values
*/

//typedef char bool;

inline int MAX(const int a, const int b){
  return b > a ? b : a;
}

inline int MIN(const int a, const int b){
  return b < a ? b : a;
}

void SWAP(double *i, double *j) {
  double t = *i;
  *i = *j;
  *j = t;
  return;
}
