/*Copyright (c) 2017 Guillermo Navas-Palencia

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.*/


/**
 * Algorithm for the numerical evaluation of the generalized exponential
 * integral for v > 0 and x > 0 in double floating-point arithmetic
 */

#ifndef EXPINT_H_
#define EXPINT_H_

// series
double expint_series_a(const double v, const double x);
double expint_series_b(const double v, const double x);
double expint_series_n(const int n, const double x);
double expint_laguerre_series(const double v, const double x);
double expint_series_n_x_le_2(const int n, const double x);
double expint_n_e_acc(const int n, const double e, const double x);

// exponential integral
double expintei(const double x);
double expinte1(const double x);

// asymptotic expansions
int use_expint_asymp_v(const double v, const double x);
int use_expint_asymp_x(const double v, const double x);
double expint_asymp_x(const double v, const double x, int n);
double expint_asymp_v(const double v, const double x, int n);

// generalized exponential integral
double expint_n(const int n, const double x);
double expint_v(const double v, const double x);
double expint_v_imp(const int n, const double e, const double x);

#endif /* EXPINT_H_ */
