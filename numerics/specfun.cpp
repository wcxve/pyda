#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "expint/expint.hpp"

namespace py = pybind11;

//#include <boost/math/special_functions/hypergeometric_1F1.hpp>
//static inline double hyp1f1(double a, double b, double x) {
//    return boost::math::hypergeometric_1F1(a, b, x);
//}
//void cutoffpl(double alpha, double beta, py::array_t<double> ebins, py::array_t<double> flux) {
//    auto _ebins = ebins.unchecked<1>();
//    auto _flux = flux.mutable_unchecked<1>();
//
//    const double one_minus_alpha = 1.0 - alpha;
//    const double two_minus_alpha = 2.0 - alpha;
//    double ei = _ebins[0];
//    double xi = ei/beta;
//    double integral_low = pow(ei, one_minus_alpha)/one_minus_alpha * exp(-xi)*hyp1f1(1.0, two_minus_alpha, xi);
//    double integral_high = 0.0;
//    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
//        ei = _ebins[i];
//        xi = ei/beta;
//        integral_high = pow(ei, one_minus_alpha)/one_minus_alpha * exp(-xi)*hyp1f1(1.0, two_minus_alpha, xi);
//        _flux[i - 1] = integral_high - integral_low;
//        integral_low = integral_high;
//    }
//}
//void cutoffpl(double alpha, double beta, py::array_t<double> ebins, py::array_t<double> flux) {
//    auto _ebins = ebins.unchecked<1>();
//    auto _flux = flux.mutable_unchecked<1>();
//
//    const double one_minus_alpha = 1.0 - alpha;
//    const double two_minus_alpha = 2.0 - alpha;
//    double ei = _ebins[0];
//    double xi = -ei/beta;
//    double integral_low = pow(ei, one_minus_alpha)/one_minus_alpha * hyp1f1(one_minus_alpha, two_minus_alpha, xi);
//    double integral_high = 0.0;
//    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
//        ei = _ebins[i];
//        xi = -ei/beta;
//        integral_high = pow(ei, one_minus_alpha)/one_minus_alpha * hyp1f1(one_minus_alpha, two_minus_alpha, xi);
//        _flux[i - 1] = integral_high - integral_low;
//        integral_low = integral_high;
//    }
//}
//#include <gsl/gsl_sf_gamma.h>
//static inline double gammainc(double a, double x) {
//    return gsl_sf_gamma_inc(a, x);
//}
//void cutoffpl(double alpha, double beta, py::array_t<double> ebins, py::array_t<double> flux) {
//    auto _ebins = ebins.unchecked<1>();
//    auto _flux = flux.mutable_unchecked<1>();
//
//    const double one_minus_alpha = 1.0 - alpha;
//    const double multiplier = pow(beta, one_minus_alpha);
//    double integral_low = gammainc(one_minus_alpha, _ebins[0]/beta);
//    double integral_high = 0.0;
//    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
//        integral_high = gammainc(one_minus_alpha, _ebins[i]/beta);
//        _flux[i - 1] = multiplier*(integral_low - integral_high);
//        integral_low = integral_high;
//    }
//}

//void cutoffpl_inplace(double alpha, double beta, py::array_t<double> ebins, py::array_t<double> flux) {
//    auto _ebins = ebins.unchecked<1>();
//    auto _flux = flux.mutable_unchecked<1>();
//
//    const double one_minus_alpha = 1.0 - alpha;
//    double ei = _ebins[0];
//    double xi = ei/beta;
//    double integral_low = pow(ei, one_minus_alpha) * expint_v(alpha, xi);
//    double integral_high;
//    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
//        ei = _ebins[i];
//        xi = ei/beta;
//        integral_high = pow(ei, one_minus_alpha) * expint_v(alpha, xi);
//        _flux[i - 1] = integral_low - integral_high;
//        integral_low = integral_high;
//    }
//}

py::array_t<double> cutoffpl(double alpha, double beta, py::array_t<double> ebins) {
    auto _ebins = ebins.unchecked<1>();
    auto flux = py::array_t<double>(ebins.size() - 1);
    py::buffer_info flux_buf = flux.request();
    double *flux_ptr = static_cast<double *>(flux_buf.ptr);

    const double one_minus_alpha = 1.0 - alpha;
    const double inv_beta = 1.0/beta;
    double ei = _ebins[0];
    double xi = inv_beta*ei;
    double integral_low = pow(ei, one_minus_alpha) * expint_v(alpha, xi);
    double fi, integral_high;
    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
        ei = _ebins[i];
        xi = inv_beta*ei;
        integral_high = pow(ei, one_minus_alpha) * expint_v(alpha, xi);
        fi = integral_low - integral_high;
        flux_ptr[i - 1] = (fi >= 0.0) ? fi : abs(fi);
        integral_low = integral_high;
    }

    return flux;
}

py::array_t<double> cutoffpl_dalpha(double alpha, double beta, py::array_t<double> ebins, double delta=0.0001) {
    auto _ebins = ebins.unchecked<1>();
    auto dalpha = py::array_t<double>(ebins.size() - 1);
    py::buffer_info dalpha_buf = dalpha.request();
    double *dalpha_ptr = static_cast<double *>(dalpha_buf.ptr);

    const double step = delta;//abs(delta*alpha);
    const double step_2 = 2.0*step;
    const double alpha_high = alpha + step;
    const double alpha_low = alpha - step;
    const double one_minus_alpha_high = 1.0 - alpha_high;
    const double one_minus_alpha_low = 1.0 - alpha_low;
    const double inv_beta = 1.0/beta;
    double ei = _ebins[0];
    double xi = inv_beta*ei;
    double integral_high_ah, integral_high_al;
    double integral_low_ah = pow(ei, one_minus_alpha_high) * expint_v(alpha_high, xi);
    double integral_low_al = pow(ei, one_minus_alpha_low) * expint_v(alpha_low, xi);
    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
        ei = _ebins[i];
        xi = inv_beta*ei;
        integral_high_ah = pow(ei, one_minus_alpha_high) * expint_v(alpha_high, xi);
        integral_high_al = pow(ei, one_minus_alpha_low) * expint_v(alpha_low, xi);
        dalpha_ptr[i - 1] = ((integral_low_ah - integral_high_ah) - (integral_low_al - integral_high_al))/step_2;
        integral_low_ah = integral_high_ah;
        integral_low_al = integral_high_al;
    }

    return dalpha;
}

//#include "boost/math/special_functions/hypergeometric_pFq.hpp"
//static inline double hyp_ppfqq(double p, double q, double x) {
//    return boost::math::hypergeometric_pFq({p, p}, {q, q}, x);
//}
//
//py::array_t<double> cutoffpl_dalpha(double alpha, double beta, py::array_t<double> ebins) {
//    auto _ebins = ebins.unchecked<1>();
//    auto dalpha = py::array_t<double>(ebins.size() - 1);
//    py::buffer_info dalpha_buf = dalpha.request();
//    double *dalpha_ptr = static_cast<double *>(dalpha_buf.ptr);
//
//    const double one_minus_alpha = 1.0 - alpha;
//    const double inv_one_minus_alpha2 = 1.0/(one_minus_alpha*one_minus_alpha);
//    const double two_minus_alpha = 2.0 - alpha;
//    const double beta_one_minus_alpha = pow(beta, one_minus_alpha);
//    const double inv_beta = 1.0/beta;
//
//    double ei = _ebins[0];
//    double xi = inv_beta*ei;
//    double f_high;
//    double f_low = inv_one_minus_alpha2*pow(ei, one_minus_alpha)*hyp_ppfqq(one_minus_alpha, two_minus_alpha, -xi) \
//                   - beta_one_minus_alpha*log(ei)*(tgamma(one_minus_alpha) - pow(xi, one_minus_alpha)*expint_v(alpha, xi));
//    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
//        ei = _ebins[i];
//        xi = inv_beta*ei;
//        f_high = inv_one_minus_alpha2*pow(ei, one_minus_alpha)*hyp_ppfqq(one_minus_alpha, two_minus_alpha, -xi) \
//                 - beta_one_minus_alpha*log(ei)*(tgamma(one_minus_alpha) - pow(xi, one_minus_alpha)*expint_v(alpha, xi));
//        dalpha_ptr[i - 1] = f_high - f_low;
//        f_low = f_high;
//    }
//
//    return dalpha;
//}

py::array_t<double> cutoffpl_dbeta(double alpha, double beta, py::array_t<double> ebins) {
    auto _ebins = ebins.unchecked<1>();
    auto dbeta = py::array_t<double>(ebins.size() - 1);
    py::buffer_info dbeta_buf = dbeta.request();
    double *dbeta_ptr = static_cast<double *>(dbeta_buf.ptr);

    const double alpha_minus_one = alpha - 1.0;
    const double two_minus_alpha = 2.0 - alpha;
    const double inv_beta = 1.0/beta;
    const double inv_beta2 = inv_beta*inv_beta;
    double ei = _ebins[0];
    double xi = inv_beta*ei;
    double fi, integral_high;
    double integral_low = pow(ei, two_minus_alpha) * expint_v(alpha_minus_one, xi);
    for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
        ei = _ebins[i];
        xi = inv_beta*ei;
        integral_high = pow(ei, two_minus_alpha) * expint_v(alpha_minus_one, xi);
        fi = inv_beta2*(integral_low - integral_high);
        dbeta_ptr[i - 1] = (fi >= 0.0) ? fi : abs(fi);
        integral_low = integral_high;
    }

    return dbeta;
}


PYBIND11_MODULE(specfun, module) {
    module.doc() = "Spectral function library";
    //module.def("hyp1f1", &hyp1f1, "Confluent hypergeometric function");
    //module.def("icutoffpl", &cutoffpl_inplace, "Cut-off powerlaw integrated over x (flux inplace)");
    module.def("cutoffpl", &cutoffpl, "Cut-off powerlaw integrated over x");
    module.def("cutoffpl_dalpha", &cutoffpl_dalpha, "Partial alpha (central difference approximation) of cut-off powerlaw integrated over x",
               py::arg("alpha"), py::arg("beta"), py::arg("ebins"), py::arg("delta")=0.0001);
    //module.def("cutoffpl_dalpha", &cutoffpl_dalpha, "Partial alpha of cut-off powerlaw integrated over x",
    //           py::arg("alpha"), py::arg("beta"), py::arg("ebins"), py::arg("delta")=0.0001);
    module.def("cutoffpl_dbeta", &cutoffpl_dbeta, "Partial beta of cut-off powerlaw integrated over x");
    module.def("expint", &expint_v);
}