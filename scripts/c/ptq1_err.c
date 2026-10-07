// Compare PTQ1_0 absmax vs fit: relative MSE on Gaussian and Laplace rows, and round-trip of exact ternary rows.
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static float gauss(void){ double u=(rand()+1.0)/(RAND_MAX+2.0), v=(rand()+1.0)/(RAND_MAX+2.0); return sqrt(-2*log(u))*cos(6.2831853*v); }
static float lap(void){ double u=(rand()+1.0)/(RAND_MAX+2.0)-0.5; return (u<0?1:-1)*log(1-2*fabs(u)); }
static double run(const float*x,int n){
  size_t rs=ggml_row_size(GGML_TYPE_PTQ1_0,n); void*q=malloc(rs); float*y=malloc(n*4);
  ggml_quantize_chunk(GGML_TYPE_PTQ1_0,x,q,0,1,n,NULL);
  ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float(q,y,n);
  double e=0,s=0; for(int i=0;i<n;i++){e+=(x[i]-y[i])*(x[i]-y[i]);s+=x[i]*x[i];}
  free(q);free(y); return e/s;
}
int main(void){
  int n=128*4096; float*x=malloc(n*4);
  for(int i=0;i<n;i++)x[i]=0.02f*gauss(); printf("gauss   rel_mse %.4f\n",run(x,n));
  for(int i=0;i<n;i++)x[i]=0.02f*lap();   printf("laplace rel_mse %.4f\n",run(x,n));
  for(int i=0;i<n;i++)x[i]=0.03f*((rand()%3)-1); printf("ternary rel_mse %.6f\n",run(x,n));
  return 0;
}
