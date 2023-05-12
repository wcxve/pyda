// https://github.com/Expander/polylogarithm

/**
 * @brief Real dilogarithm \f$\operatorname{Li}_2(x)\f$
 * @param x real argument
 * @return \f$\operatorname{Li}_2(x)\f$
 * @author Alexander Voigt
 *
 * Implemented as a rational function approximation with a maximum
 * error of 5e-17
 * [[arXiv:2201.01678](https://arxiv.org/abs/2201.01678)].
 */
double Li2(double x) noexcept {
   const double PI = 3.1415926535897932;
   const double P[] = {
      0.9999999999999999502e+0,
     -2.6883926818565423430e+0,
      2.6477222699473109692e+0,
     -1.1538559607887416355e+0,
      2.0886077795020607837e-1,
     -1.0859777134152463084e-2
   };
   const double Q[] = {
      1.0000000000000000000e+0,
     -2.9383926818565635485e+0,
      3.2712093293018635389e+0,
     -1.7076702173954289421e+0,
      4.1596017228400603836e-1,
     -3.9801343754084482956e-2,
      8.2743668974466659035e-4
   };

   double y = 0, r = 0, s = 1;

   // transform to [0, 1/2]
   if (x < -1) {
      const double l = std::log(1 - x);
      y = 1/(1 - x);
      r = -PI*PI/6 + l*(0.5*l - std::log(-x));
      s = 1;
   } else if (x == -1) {
      return -PI*PI/12;
   } else if (x < 0) {
      const double l = std::log1p(-x);
      y = x/(x - 1);
      r = -0.5*l*l;
      s = -1;
   } else if (x == 0) {
      return 0;
   } else if (x < 0.5) {
      y = x;
      r = 0;
      s = 1;
   } else if (x < 1) {
      y = 1 - x;
      r = PI*PI/6 - std::log(x)*std::log(y);
      s = -1;
   } else if (x == 1) {
      return PI*PI/6;
   } else if (x < 2) {
      const double l = std::log(x);
      y = 1 - 1/x;
      r = PI*PI/6 - l*(std::log(y) + 0.5*l);
      s = 1;
   } else {
      const double l = std::log(x);
      y = 1/x;
      r = PI*PI/3 - 0.5*l*l;
      s = -1;
   }

   const double y2 = y*y;
   const double y4 = y2*y2;
   const double p = P[0] + y * P[1] + y2 * (P[2] + y * P[3]) +
                    y4 * (P[4] + y * P[5]);
   const double q = Q[0] + y * Q[1] + y2 * (Q[2] + y * Q[3]) +
                    y4 * (Q[4] + y * Q[5] + y2 * Q[6]);

   return r + s*y*p/q;
}

// Li_3(x) for x in [-1,0]
double li3_neg(double x) noexcept {
    const double cp[] = {
        0.9999999999999999795e+0, -2.0281801754117129576e+0,
        1.4364029887561718540e+0, -4.2240680435713030268e-1,
        4.7296746450884096877e-2, -1.3453536579918419568e-3
    };
    const double cq[] = {
        1.0000000000000000000e+0, -2.1531801754117049035e+0,
        1.6685134736461140517e+0, -5.6684857464584544310e-1,
        8.1999463370623961084e-2, -4.0756048502924149389e-3,
        3.4316398489103212699e-5
    };

    const double x2 = x*x;
    const double x4 = x2*x2;
    const double p = cp[0] + x*cp[1] + x2*(cp[2] + x*cp[3]) + x4*(cp[4] + x*cp[5]);
    const double q = cq[0] + x*cq[1] + x2*(cq[2] + x*cq[3]) + x4*(cq[4] + x*cq[5] + x2*cq[6]);
    return x*p/q;
}

// Li_3(x) for x in [0,1/2]
double li3_pos(double x) noexcept {
   const double cp[] = {
      0.9999999999999999893e+0, -2.5224717303769789628e+0,
      2.3204919140887894133e+0, -9.3980973288965037869e-1,
      1.5728950200990509052e-1, -7.5485193983677071129e-3
   };
   const double cq[] = {
      1.0000000000000000000e+0, -2.6474717303769836244e+0,
      2.6143888433492184741e+0, -1.1841788297857667038e+0,
      2.4184938524793651120e-1, -1.8220900115898156346e-2,
      2.4927971540017376759e-4
   };

   const double x2 = x*x;
   const double x4 = x2*x2;
   const double p = cp[0] + x*cp[1] + x2*(cp[2] + x*cp[3]) + x4*(cp[4] + x*cp[5]);
   const double q = cq[0] + x*cq[1] + x2*(cq[2] + x*cq[3]) + x4*(cq[4] + x*cq[5] + x2*cq[6]);
   return x*p/q;
}


/**
 * @brief Real trilogarithm \f$\operatorname{Li}_3(x)\f$
 * @param x real argument
 * @return \f$\operatorname{Li}_3(x)\f$
 * @author Alexander Voigt
 */
 /* apdapt for x in [0, 1] */
double Li3(double x) noexcept
{
    const double zeta2 = 1.6449340668482264;
    const double zeta3 = 1.2020569031595943;

    if (x < 0.5) {
          return li3_pos(x);
    } else {
        const double l = std::log(x);
        return -li3_neg(1 - 1/x) - li3_pos(1 - x) + zeta3 + l*(zeta2 + l*(-0.5*std::log(1 - x) + 1.0/6*l));
   }
}