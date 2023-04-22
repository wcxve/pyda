%module expint
%{
#include "../../expint/include/expint.h"
#include "../../expint/include/expint_const.h"
%}

extern double expint_n(const int n, const double x);
extern double expint_v(const double v, const double x);
extern double expint_v_imp(const int n, const double e, const double x);
