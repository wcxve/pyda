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

#ifndef EXPINT_HPP
#define EXPINT_HPP

#include <cmath>
#include <limits>
// #include <quadmath.h>
#include <vector>

// mathematical constants
# define SQRT_PI      1.7724538509055160273
# define EULER_MASC   0.5772156649015328606

// machine precision
# define EPSILON      1.11022302462516e-16
# define INF          std::numeric_limits<double>::infinity()

// series expansions
double expint_series_a(const double v, const double x);
double expint_series_b(const double v, const double x);
double expint_series_n(const int n, const double x);
double expint_laguerre_series(const double v, const double x);
double expint_series_n_x_le_2(const int n, const double x);
// double expint_n_e_acc(const int n, const double e, const double x);

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
// double expint_v_imp(const int n, const double e, const double x);

/**
 * Series expansions
 */

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

// double expint_n_e_acc(const int n, const double e, const double x)
// {
//   int k, terms = 0;
//   double b;
//    __float128 t, u, s, ss, q, d, r;
//    __float128 one = 1.0q;

//   expint_series1_terms(n+e, x, &b, &terms);

//   t = 1 - n;
//   t -= e;
//   u = expq(lgammaq(t)); //tgammaq returns incorrect sign for very small e
//   if (n % 2 == 0)
//   {
//     if (e < 0.0q)
//       u = -fabsq(u);
//   }
//   else
//   {
//     if (e > 0.0q)
//       u = -fabsq(u);
//   }

//   u *= powq(x, -t);

//   s = one / t;
//   q = d = one;
//   for (k = 1; k <= terms; k++)
//   {
//     q *= -x;
//     d *= k;
//     r = q / (d * (t + k));
//     s += r;
//   }
//   ss = u - s;
//   return static_cast<double>(ss);
// }


/**
 * Exponential integral E1(x) and Ei(x).
 * Reference:  W. J. Cody and H. C. Thacher, Jr. {\em Chebyshev approximation
 * for the exponential integral $E_i(x)$}. Math. Comp. 23, pp. 289-303, (1969).
 */

static const double A[7] = {
  1.1669552669734461083368e2,
  2.1500672908092918123209e3,
  1.5924175980637303639884e4,
  8.9904972007457256553251e4,
  1.5026059476436982420737e5,
 -1.4815102102575750838086e5,
  5.0196785185439843791020e0
};

static const double B[6] = {
  4.0205465640027706061433e1,
  7.5043163907103936624165e2,
  8.1258035174768735759855e3,
  5.2440529172056355429883e4,
  1.8434070063353677359298e5,
  2.5666493484897117319268e5
};

static const double C[9] = {
  3.828573121022477169108e-1,
  1.107326627786831743809e+1,
  7.246689782858597021199e+1,
  1.700632978311516129328e+2,
  1.698106763764238382705e+2,
  7.633628843705946890896e+1,
  1.487967702840464066613e+1,
  9.999989642347613068437e-1,
  1.737331760720576030932e-8
};

static const double D[9] = {
  8.258160008564488034698e-2,
  4.344836335509282083360e+0,
  4.662179610356861756812e+1,
  1.775728186717289799677e+2,
  2.953136335677908517423e+2,
  2.342573504717625153053e+2,
  9.021658450529372642314e+1,
  1.587964570758947927903e+1,
  1.000000000000000000000e+0
};

static const double Em4[10] = {
  1.3276881505637444622987e+2,
  3.5846198743996904308695e+4,
  1.7283375773777593926828e+5,
  2.6181454937205639647381e+5,
  1.7503273087497081314708e+5,
  5.9346841538837119172356e+4,
  1.0816852399095915622498e+4,
  1.0611777263550331766871e03,
  5.2199632588522572481039e+1,
  9.9999999999999999087819e-1
};

static const double Fm4[10] = {
  3.9147856245556345627078E+4,
  2.5989762083608489777411E+5,
  5.5903756210022864003380E+5,
  5.4616842050691155735758E+5,
  2.7858134710520842139357E+5,
  7.9231787945279043698718E+4,
  1.2842808586627297365998E+4,
  1.1635769915320848035459E+3,
  5.4199632588522559414924E+1,
  1.000000000000000000000e+0
};

static const double PLG[4] = {
  -2.4562334077563243311E+01,
  2.3642701335621505212E+02,
  -5.4989956895857911039E+02,
  3.5687548468071500413E+02
};

static const double QLG[4] = {
  -3.5553900764052419184E+01,
  1.9400230218539473193E+02,
  -3.3442903192607538956E+02,
  1.7843774234035750207E+02
};

static const double P06[10] = {
  -1.2963702602474830028590E01,
  -1.2831220659262000678155E03,
  -1.4287072500197005777376E04,
  -1.4299841572091610380064E06,
  -3.1398660864247265862050E05,
  -3.5377809694431133484800E08,
  3.1984354235237738511048E08,
  -2.5301823984599019348858E10,
  1.2177698136199594677580E10,
  -2.0829040666802497120940E11
};

static const double Q06[10] = {
  7.6886718750000000000000E01,
  -5.5648470543369082846819E03,
  1.9418469440759880361415E05,
  -4.2648434812177161405483E06,
  6.4698830956576428587653E07,
  -7.0108568774215954065376E08,
  5.4229617984472955011862E09,
  -2.8986272696554495342658E10,
  9.8900934262481749439886E10,
  -8.9673749185755048616855E10
};

static const double R612[10] = {
  -2.645677793077147237806E00,
  -2.378372882815725244124E00,
  -2.421106956980653511550E01,
  1.052976392459015155422E01,
  1.945603779539281810439E01,
  -3.015761863840593359165E01,
  1.120011024227297451523E01,
  -3.988850730390541057912E00,
  9.565134591978630774217E00,
  9.981193787537396413219E-1
};

static const double S612[9] = {
  1.598517957704779356479E-4,
  4.644185932583286942650E00,
  3.697412299772985940785E02,
  -8.791401054875438925029E00,
  7.608194509086645763123E02,
  2.852397548119248700147E01,
  4.731097187816050252967E02,
  -2.369210235636181001661E02,
  1.249884822712447891440E00
};

static const double P1224[10] = {
  -1.647721172463463140042E00,
  -1.860092121726437582253E01,
  -1.000641913989284829961E01,
  -2.105740799548040450394E01,
  -9.134835699998742552432E-1,
  -3.323612579343962284333E01,
  2.495487730402059440626E01,
  2.652575818452799819855E01,
  -1.845086232391278674524E00,
  9.999933106160568739091E-1
};

static const double Q1224[9] = {
  9.792403599217290296840E01,
  6.403800405352415551324E01,
  5.994932325667407355255E01,
  2.538819315630708031713E02,
  4.429413178337928401161E01,
  1.192832423968601006985E03,
  1.991004470817742470726E02,
  -1.093556195391091143924E01,
  1.001533852045342697818E00
};

static const double P24[10] = {
  1.75338801265465972390E02,
  -2.23127670777632409550E02,
  -1.81949664929868906455E01,
  -2.79798528624305389340E01,
  -7.63147701620253630855E00,
  -1.52856623636929636839E01,
  -7.06810977895029358836E00,
  -5.00006640413131002475E00,
  -3.00000000320981265753E00,
  1.00000000000000485503E00
};

static const double Q24[9] = {
  3.97845977167414720840E04,
  3.97277109100414518365E00,
  1.37790390235747998793E02,
  1.17179220502086455287E02,
  7.04831847180424675988E01,
  -1.20187763547154743238E01,
  -7.99243595776339741065E00,
  -2.99999894040324959612E00,
  1.99999999999048104167E00
};

double expintei(const double x) {
  double ans;
  double frac, t, y, y2, w, sump, sumq;
  double xx0, xmx0;

  std::vector<double> px (10);
  std::vector<double> qx (10);

  double xmax = 716.351;
  double x0 = 3.7250741078136663466e-1;
  double x01 = 381.5;
  double x11 = 1024.0;
  double x02 = -5.1182968633365538008e-5;

  if (x < 0.0) {
    y = std::abs(x);
    if (y <= 1.0) {
      sump = A[6] * y + A[0];
      sumq = y + B[0];
      for (int k = 1; k <= 5; k++) {
        sump = sump * y + A[k];
        sumq = sumq * y + B[k];
      }
      ans = log(y) - sump / sumq;
    } else if (y <= 4) {
      w = 1 / y;
      sump = C[0];
      sumq = D[0];
      for (int k = 1; k <= 8; k++) {
        sump = sump * w + C[k];
        sumq = sumq * w + D[k];
      }
      ans = -sump * exp(-y) / sumq;
    } else {
      w = 1 / y;
      sump = Em4[0];
      sumq = Fm4[0];
      for (int k = 1; k <= 9; k++) {
        sump = sump * w + Em4[k];
        sumq = sumq * w + Fm4[k];
      }
      ans = -w * exp(-y) * (1.0 - w * sump / sumq);
    }
  } else if (x < 6.0) {
    t = 0.6666666666666666667 * x - 2.0;
    px[0] = 0.0;
    qx[0] = 0.0;
    px[1] = P06[0];
    qx[1] = Q06[0];
    for (int k = 1; k <= 8; k++) {
      px[k + 1] = t * px[k] - px[k - 1] + P06[k];
      qx[k + 1] = t * qx[k] - qx[k - 1] + Q06[k];
    }
    sump = 0.5 * t * px[9] - px[8] + P06[9];
    sumq = 0.5 * t * qx[9] - qx[8] + Q06[9];
    frac = sump / sumq;
    xmx0 = (x - x01 / x11) - x02;
    if (std::abs(xmx0) >= 0.037) {
      ans = log(x / x0) + xmx0 * frac;
    } else {
      xx0 = x + x0;
      y = xmx0 / (xx0);
      y2 = y * y;
      sump = PLG[0];
      sumq = y2 + QLG[0];
      for (int k = 1; k <= 3; k++) {
        sump = sump * y2 + PLG[k];
        sumq = sumq * y2 + QLG[k];
      }
      ans = (sump / (sumq * (xx0)) + frac) * xmx0;
    }
  } else if (x < 12.0) {
    frac = 0.0;
    for (int k = 0; k <= 8; k++) {
      frac = S612[k] / (R612[k] + x + frac);
    }
    ans = exp(x) * (R612[9] + frac) / x;
  } else if (x < 24.0) {
    frac = 0.0;
    for (int k = 0; k <= 8; k++) {
      frac = Q1224[k] / (P1224[k] + x + frac);
    }
    ans = exp(x) * (P1224[9] + frac) / x;
  } else {
    if (x >= xmax) {
      ans = INF;
    } else {
      y = 1 / x;
      frac = 0.0;
      for (int k = 0; k <= 8; k++) {
        frac = Q24[k] / (P24[k] + x + frac);
      }
      frac += P24[9];
      ans = y + y * y * frac;
      if (x <= xmax - 24.0) {
        ans *= exp(x);
      } else {
        // reformulated to avoid premature overflow
        ans = (ans * exp(x - 40.0)) * 2.3538526683701998541e17;
      }
    }
  }
  return ans;
}


double expinte1(const double x) {
    return -expintei(-x);
}


/**
 * Asymptotic expansions for large x and v
 */

int use_expint_asymp_x(const double v, const double x)
{
  int i, n_max;
  double r;

  n_max = (int) ceil(x - v);

  r = 1.0;
  for (i = 1; i <= n_max; i++)
  {
    r *= (v + i - 1) / x;
    if (r < EPSILON)
      return i;
  }
  return 0;
}

double expint_asymp_x(const double v, const double x, const int n)
{
  int i;
  double d, s, u;

  double v1 = v - 1.0;

  // series
  u = 1.0;
  d = x;
  s = 1.0 / x;

  for (i = 1; i <= n; i++)
  {
    u *= -(v1 + i);
    d *= x;
    s += u / d;
  }
  return exp(-x) * s;
}


int use_expint_asymp_v(const double v, const double x)
{
  int i, n_max;
  double r;

  n_max = (int) ceil(abs(v - x) - 1);

  r = 1.0 / (v - 1.0);
  for (i = 1; i <= n_max; i++) {
    r *= x / (v - 1.0 - i);
    if (abs(r) < EPSILON)
      return i;
  }
  return 0;
}


double expint_asymp_v(const double v, const double x, const int n)
{
  int i;
  double d, s, u;

  // series
  u = 1.0 / (v - 1.0);
  s = u;
  d = 1.0;

  for (i = 1; i <= n; i++)
  {
    u /= (v - 1.0 - i);
    d *= -x;
    s += u * d;
  }
  return s * exp(-x);
}


/**
 * Special cases for generalized exponential integral with integer nu.
 * iflag return 1 if special case is detected, otherwise 0.
 */
void expint_n_special_cases(const int n, const double x, int *iflag,
    double *result)
{
  // x == 0
  if (x == 0.0)
  {
    if (n <= 1) // exponential integral is not defined
      *result = INF;
    else
      *result =  1.0 / (n - 1);
    *iflag = 1;
  }

  // case n == 0
  if (n == 0)
  {
    *result = exp(-x) / x;
    *iflag = 1;
  }

  // case n == 1
  if (n == 1)
  {
    if (x > 0.9 && x < 10.0)
    {
      *result = expinte1(x);
      *iflag = 1;
    }
  }
}

/**
 * Special cases for generalized exponential integral with double nu.
 * iflag return 1 if special case is detected, otherwise 0.
 */
void expint_v_special_cases(const double v, const double x, int *iflag,
    double *result)
{
  double sqrt_x;

  // x == 0
  if (x == 0.0)
  {
    if (v <= 1.0)   // exponential integral is not defined
      *result = INF;
    else
      *result = 1.0 / (v - 1.0);
    *iflag = 1;
  }

  // case v = 1/2
  if (v == 0.5 && x < 10.0)
  {
    // x >= 10 some loss of digits is observed
    sqrt_x = sqrt(x);
    *result = SQRT_PI * erfc(sqrt_x) / sqrt_x;
    *iflag = 1;
  }
}

/**
 * Series expansion for small x (<= 1.5) and double v
 */
double expint_small_x(const double v, const double x)
{
  if (v / x > 10.0) // fast convergence
    return expint_series_a(v, x);

  if (v > 1.5 && x > 0.5) // slow but accurate
    return expint_laguerre_series(v, x);
  else
  {
    if (v < 0.9) // all terms of the expansion are positive
      return expint_series_b(v, x);
    else
      return expint_series_a(v, x);
  }
}

/**
 * Asymptotic expansion or Laguerre series for large v
 */
double expint_large_v(const double v, const double x)
{
  int iter;

  // fixed and small x and large v
  if (x < 5.0)
  {
    iter = use_expint_asymp_v(v, x);
    if (iter)
      return expint_asymp_v(v, x, iter);
    else
      expint_laguerre_series(v, x);
  }
  return expint_laguerre_series(v, x);
}

/**
 * Asymptotic expansion or Laguerre series for large x
 */
double expint_large_x(const double v, const double x)
{
  int iter;

  // fixed v and large v
  if (x / abs(v) > 100.0) {
    iter = use_expint_asymp_x(v, x);
    if (iter)
      return expint_asymp_x(v, x, iter);
    else
      return expint_laguerre_series(v, x);
  }
  else
    return expint_laguerre_series(v, x);
}

/**
 * Algorithm for the generalized exponential integral with integer nu
 */
double expint_n(const int n, const double x)
{
  int iflag = 0;
  double result = 0.0;

  // special cases
  expint_n_special_cases(n, x, &iflag, &result);
  if (iflag) return result;

  // small x
  if (x <= 1.5 && n < 20)
    return expint_series_n(n, x);
  else if (x <= 2.0 && n <= 10)
    // use series expansion in terms of E1(x) to avoid many iterations
    // laguerre series
    return expint_series_n_x_le_2(n, x);
  else if (n >= x)
    return expint_large_v(n, x);
  else
    return expint_large_x(n, x);
}

/**
 * Algorithm for the generalized exponential integral with double nu
 */
double expint_v(const double v, const double x)
{
  int iflag = 0, vint;
  double result = 0.0;

  // v is integer
  vint = int(v);
  if (v > 0.0 && v == vint)
    return expint_n(vint, x);

  // special cases
  expint_v_special_cases(v, x, &iflag, &result);
  if (iflag) return result;

  // general cases
  if (x <= 1.0)
    return expint_small_x(v, x);
  else if (abs(v) >= x)
    return expint_large_v(v, x);
  else
    return expint_large_x(v, x);
}

// /**
//  * Algorithm for the generalized exponential integral with double nu
//  * accuracy improved: v split into an integral and decimal fractional part.
//  */
// double expint_v_imp(const int n, const double e, const double x)
// {
//   int iflag = 0;
//   double result, v;
//   result = v = 0.0;

//   // check if input is integer, return expint_n
//   if (e == 0.0)
//     return expint_n(n, x);

//   // special cases
//   if (std::abs(e) < 0.1 && (n < 10 && n > 0) && x < 2.0)
//     return expint_n_e_acc(n, e, x);

//   expint_v_special_cases(v, x, &iflag, &result);
//   if (iflag) return result;

//   // general cases
//   v = n + e;
//   if (x <= 1.0)
//     return expint_small_x(v, x);
//   else if (v >= x)
//     return expint_large_v(v, x);
//   else
//     return expint_large_x(v, x);
// }

#endif /* EXPINT_HPP */