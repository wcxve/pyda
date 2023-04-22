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
 * Series expansions
 */

#include <cmath>
#include <quadmath.h>

#include "expint_const.h"
#include "expint.h"

// choose n such that the bound of the remainder of the series is less than
// the threshold = 2^(-53). Perform linear search (to be optimised).
int expint_series1_terms(const double v, const double x, double* bound, int* n) 
{
  int n_max = 50;
  double b, t, u, w;

  u = 1.0;
  t = 1.0 - v;
  w = 1.0;

  for (int i = 1; i < n_max; i++) 
  {
    u *= x;
    t += 1;
    w *= i;
    b = u / (t * w);

    if (std::abs(b) < EPSILON) 
    {
      *n = i;
      *bound = b;
      return 1;
    }
  }
  return 0;
}

// choose n for series 2. Perform linear search
int expint_series2_terms(const double v, const double x, double* bound, int* n) 
{
  int n_max = 50;
  double b, t, u, w;

  // first iteration
  u = 1.0;
  t = 1.0 - v;
  w = 1.0;

  for (int i = 1; i < n_max; i++) 
  {
    u *= x;
    w *= (t + i);
    b = u / w;

    if (std::abs(b) < EPSILON) 
    {
      *n = i;
      *bound = b;
      return 1;
    }
  }
  return 0;
}


// construct error bound for remainder (JF)
double _expint_series1_bound_jf(const double v, const double x, const double b,
    const int n) 
{
  double c, d, u;

  u = std::abs(2.0 - v + n);
  d = x * (1.0 + 1.0 / u) / u;

  if (d < 1) 
  {
    c = 1.0 / (1.0 - d);
    return c * b;
  } 
  else 
    return INF;
}

// contruct error bound from 2F2 functional identity (GNP)
double _expint_series1_bound_gnp(const double v, const double x, const double b,
    const int n) 
{
  double c, u;

  u = std::abs(1.0 - v + n);
  c = 1 + u * x / ((1.0 + n) * (u + 1.0));
  return c * b;
}

// construct error bound for remainder series 2
double _expint_series2_bound(const double v, const double x, const double b,
    const int n) 
{
  double c, d, u;

  u = std::abs(2.0 - v + n);
  d = x * (1.0 + std::abs(v - 1.0) / u) / u;

  if (d < 1) 
  {
    c = 1.0 / (1.0 - d);
    return c * b;
  } 
  else 
    return INF;
}

// series expansion 1 for small x, alternating series.
double expint_series_a(const double v, const double x) 
{
  int k, n = 0;
  double b, con, d, q, r, u, shi, slo, t, thi, tlo;

  // select number of terms
  expint_series1_terms(v, x, &b, &n);

  t = 1.0 - v;
  u = tgamma(t) * pow(x, -t);

  shi = 1.0 / t;
  slo = 0.0;
  q = 1.0;
  d = 1.0;

  for (k = 1; k <= n; k++) 
  {
    q *= -x;
    d *= k;
    r = q / (d * (t + k));

    // sum double-double
    thi = shi + r;
    con = thi - shi;
    tlo = (shi - (thi - con) + (r - con));
    tlo += slo;

    shi = thi + tlo;
    slo = (thi - shi) + tlo;
  }
  shi = -shi;
  slo = -slo;
  thi = shi + u;
  con = thi - shi;
  tlo = (shi - (thi - con) + (u - con));
  tlo += slo;

  return thi + tlo;
}

double expint_series_b(const double v, const double x) 
{
  int k, n = 0;
  double aux, b, con, d, q, r, u, shi, slo, t, thi, tlo;

  // select number of terms
  expint_series2_terms(v, x, &b, &n);

  t = 1.0 - v;
  u = tgamma(t) * pow(x, -t);

  shi = 1.0;
  slo = 0.0;
  q = 1.0;
  d = 1.0;

  for (k = 1; k <= n; k++) 
  {
    q *= x;
    d *= (t + k);
    r = q / d;
    // sum double-double
    thi = shi + r;
    con = thi - shi;
    tlo = (shi - (thi - con) + (r - con));
    tlo += slo;

    shi = thi + tlo;
    slo = (thi - shi) + tlo;
  }
  aux = exp(-x) / (v - 1.0);
  shi = shi * aux;
  slo = slo * aux;
  thi = shi + u;
  con = thi - shi;
  tlo = (shi - (thi - con) + (u - con));
  tlo += slo;

  return thi + tlo;
}


// compute E_n(x), n is integer positive
// code based on cephes expn.c power series expansion
double expint_series_n(const int n, const double x)
{
  int i, k, terms = 0;
  double psi0, factor;
  double b, xk, yk, pk;
  double r, shi, slo, thi, tlo, con;

  // select number of terms
  expint_series1_terms((double)n, x, &b, &terms);

  // compute digamma function, \psi_0, using its series expansion for integer
  // argument:
  // \psi_0 = -EULER_MAS - sum_{i=1}^{n-1} 1/i
  psi0 = -EULER_MASC;
  for (i = 1; i < n; i++)
    psi0 += 1.0 / i;

  // this series is used for n < 20, therefore a direct evaluation of the
  // factor is safe: (-x) ^ (n-1) * psi / gamma(n)
  factor = pow(-x, n - 1.0) * (psi0 - log(x)) / tgamma(n);

  // series
  xk = 0.0;
  yk = 1.0;
  pk = 1.0 - n;

  if (n == 1) 
  {
    shi = 0.0;
    slo = 0.0;
  } 
  else 
  {
    shi = 1.0 / pk;
    slo = 0.0;
  }

  for (k = 0; k <= terms; k++) 
  {
    xk += 1.0;
    yk *= -x / xk;
    pk += 1.0;
    if (pk != 0.0) {
      r = yk / pk;
      thi = shi + r;
      // sum double-double
      con = thi - shi;
      tlo = (shi - (thi - con) + (r - con));
      tlo += slo;

      shi = thi + tlo;
      slo = (thi - shi) + tlo;
    }
  }
  shi = -shi;
  slo = -slo;
  thi = shi + factor;
  con = thi - shi;
  tlo = (shi - (thi - con) + (factor - con));
  tlo += slo;

  return thi + tlo;
}

// Convergent Laguerre series
double expint_laguerre_series(const double v, const double x) 
{
  double Lk, Lk1, u, d, q, r, s;
  int k, maxiter = 500;

  Lk = 1.0;
  Lk1 = x + v;
  // iteration 0
  s = 1/ Lk1;

  u = 1.0;
  d = 1.0;
  for (k = 1; k < maxiter; k++) 
  {
    u *= v + k - 1;
    d *= 1 + k;
    q = (x + 2*k + v) / (k + 1) * Lk1 - (k + v - 1) / (k + 1) * Lk;

    r = u / (d * (q * Lk1));
    s += r;

    Lk = Lk1;
    Lk1 = q;
    if (std::abs(r) < 0.1 * EPSILON)
        break;
  }
  return s * exp(-x);
}

// recursion starting with E1(x)
double expint_series_n_x_le_2(const int n, const double x) 
{
  double factor;
  int k, n1, m;
  double d, s;

  // compute E1(x)
  n1 = n - 1;

  // this series is used for n < 10, therefore a direct evaluation of the
  // factor is safe: (-x) ^ (n-1) * E1(x) / gamma(n)
  factor = pow(-x, n1) * expinte1(x) / tgamma(n);

  // series
  m = n1;
  s = 1.0 / m;
  d = 1.0;
  for (k = 1; k <= n - 2; k++) {
      m *= n1 - k;
      d *= -x;
      s += d / m;
  }
  return factor + exp(-x) * s;
}

double expint_n_e_acc(const int n, const double e, const double x)
{
  int k, terms = 0;
  double b;
   __float128 t, u, s, ss, q, d, r;
   __float128 one = 1.0q;

  expint_series1_terms(n+e, x, &b, &terms);

  // std::cout << "terms = " << terms << std::endl; 
  t = 1 - n;
  t -= e;
  u = expq(lgammaq(t)); //tgammaq returns incorrect sign for very small e
  if (n % 2 == 0)
  {
    if (e < 0.0q)
      u = -fabsq(u);  
  }
  else
  {
    if (e > 0.0q)
      u = -fabsq(u);
  }

  u *= powq(x, -t);

  s = one / t;
  q = d = one;
  for (k = 1; k <= terms; k++)
  {
    q *= -x;
    d *= k;
    r = q / (d * (t + k));
    s += r;
  }
  ss = u - s;
  return static_cast<double>(ss);
}
