#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "boost/math/special_functions/hypergeometric_1F1.hpp"
#include "lib/expint.hpp"
#include "lib/polylog.hpp"

namespace py = pybind11;

py::array_t<double> bbody(double kT, py::array_t<double> ebins) {
    auto _ebins = ebins.unchecked<1>();
    auto flux = py::array_t<double>(ebins.size() - 1);
    py::buffer_info flux_buf = flux.request();
    double *flux_ptr = static_cast<double *>(flux_buf.ptr);

    const double kT2 = kT*kT;
    const double kT3 = kT*kT2;

    // 6 order series expasion at zero, in Horner form
    // (-x^6 + t^2 (90 x^4 + t (2160 t x^2 - 720 x^3)))/(4320 t^3)
    const double threshold = 5e-2*kT;
    double E = _ebins[0];
    int n = _ebins.shape(0);
    double integral_low, integral_high;
    if (E >= threshold) {
        n = 0;
    } else {
        const double v1 = 2160.0*kT;
        const double norm_ = 8.0525/(4320*kT2*kT2*kT3);
        double E2 = E*E;
        double E3 = E2*E;
        double E4 = E2*E2;
        double E6 = E3*E3;
        integral_low = kT2*(90.0*E4 + kT*(v1*E2 - 720.0*E3)) - E6;
        for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
            E = _ebins[i];
            E2 = E*E;
            E3 = E2*E;
            E4 = E2*E2;
            E6 = E3*E3;
            integral_high = kT2*(90.0*E4 + kT*(v1*E2 - 720.0*E3)) - E6;
            flux_ptr[i - 1] = norm_*(integral_high - integral_low);
            if (E >= threshold) {
                n = i;
                break;
            }
            integral_low = integral_high;
        }
    }

    const double norm = 8.0525/kT3;
    E = _ebins[n];
    double EkT = E/kT;
    double exp_EkT = exp(-EkT);
    integral_low = E*E*log1p(-exp_EkT) - 2.0*kT*(E*Li2(exp_EkT) + kT*Li3(exp_EkT));
    for (py::ssize_t i = n + 1; i < _ebins.shape(0); i++) {
        E = _ebins[i];
        EkT = E/kT;
        exp_EkT = exp(-EkT);
        integral_high = E*E*log1p(-exp_EkT) - 2.0*kT*(E*Li2(exp_EkT) + kT*Li3(exp_EkT));
        flux_ptr[i - 1] = norm*(integral_high - integral_low);
        if (EkT > 60.0) {
            for (py::ssize_t j = i; j < _ebins.shape(0) - 1; j++) {
                flux_ptr[j] = 0.0;
            }
            break;
        }
        integral_low = integral_high;
    }

    return flux;
}

py::array_t<double> bbodyrad(double kT, py::array_t<double> ebins) {
    auto _ebins = ebins.unchecked<1>();
    auto flux = py::array_t<double>(ebins.size() - 1);
    py::buffer_info flux_buf = flux.request();
    double *flux_ptr = static_cast<double *>(flux_buf.ptr);

    // 6 order series expasion at zero, in Horner form
    // (-x^6 + t^2 (90 x^4 + t (2160 t x^2 - 720 x^3)))/(4320 t^3)
    const double threshold = 5e-2*kT;
    double E = _ebins[0];
    int n = _ebins.shape(0);
    double integral_low, integral_high;
    if (E >= threshold) {
        n = 0;
    } else {
        const double v1 = kT*kT;
        const double v2 = 2160.0*kT;
        const double norm_ = 0.0010344/(4320*kT*v1);
        double E2 = E*E;
        double E3 = E2*E;
        double E4 = E2*E2;
        double E6 = E3*E3;
        integral_low = v1*(90.0*E4 + kT*(v2*E2 - 720.0*E3)) - E6;
        for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
            E = _ebins[i];
            E2 = E*E;
            E3 = E2*E;
            E4 = E2*E2;
            E6 = E3*E3;
            integral_high = v1*(90.0*E4 + kT*(v2*E2 - 720.0*E3)) - E6;
            flux_ptr[i - 1] = norm_*(integral_high - integral_low);
            if (E >= threshold) {
                n = i;
                break;
            }
            integral_low = integral_high;
        }
    }

    const double norm = 0.0010344 * kT;
    E = _ebins[n];
    double EkT = E/kT;
    double exp_EkT = exp(-EkT);
    integral_low = E*E*log1p(-exp_EkT) - 2.0*kT*(E*Li2(exp_EkT) + kT*Li3(exp_EkT));
    for (py::ssize_t i = n + 1; i < _ebins.shape(0); i++) {
        E = _ebins[i];
        EkT = E/kT;
        exp_EkT = exp(-EkT);
        integral_high = E*E*log1p(-exp_EkT) - 2.0*kT*(E*Li2(exp_EkT) + kT*Li3(exp_EkT));
        flux_ptr[i - 1] = norm*(integral_high - integral_low);
        if (EkT > 60.0) {
            for (py::ssize_t j = i; j < _ebins.shape(0) - 1; j++) {
                flux_ptr[j] = 0.0;
            }
            break;
        }
        integral_low = integral_high;
    }

    return flux;
}

static inline double hyp1f1(double a, double b, double x) {
    return boost::math::hypergeometric_1F1(a, b, x);
}

py::array_t<double> cutoffpl(double PhoIndex, double HighECut, py::array_t<double> ebins) {
    auto _ebins = ebins.unchecked<1>();
    auto x = py::array_t<double>(ebins.size());
    py::buffer_info x_buf = x.request();
    double *x_ptr = static_cast<double *>(x_buf.ptr);
    auto flux = py::array_t<double>(ebins.size() - 1);
    py::buffer_info flux_buf = flux.request();
    double *flux_ptr = static_cast<double *>(flux_buf.ptr);

    for (py::ssize_t i = 0; i < _ebins.shape(0); i++) {
        x_ptr[i] = _ebins[i]/HighECut;
    }

    if (PhoIndex >= 0.0) {
        const double one_minus_PhoIndex = 1.0 - PhoIndex;
        double integral_low = pow(_ebins[0], one_minus_PhoIndex) * expint_v(PhoIndex, x_ptr[0]);
        double fi, integral_high;
        for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
            integral_high = pow(_ebins[i], one_minus_PhoIndex) * expint_v(PhoIndex, x_ptr[i]);
            fi = integral_low - integral_high;
            flux_ptr[i - 1] = (fi >= 0.0) ? fi : abs(fi);
            integral_low = integral_high;
        }
    }
    else {
        const double bound = PhoIndex < -1.0 ? -PhoIndex : 1.0;
        const double one_minus_PhoIndex = 1.0 - PhoIndex;
        const double two_minus_PhoIndex = 2.0 - PhoIndex;
        int n = _ebins.shape(0) - 1;

        for (py::ssize_t i = 0; i < _ebins.shape(0); i++) {
            if (x_ptr[i] >= bound) {
                if (i != n) {
                    n = i + 1;
                }
                break;
            }
        }

        double ei = _ebins[0];
        double xi = x_ptr[0];
        double integral_low = pow(ei, one_minus_PhoIndex)/one_minus_PhoIndex * exp(-xi)*hyp1f1(1.0, two_minus_PhoIndex, xi);
        double fi, integral_high;
        for (py::ssize_t i = 1; i <= n; i++) {
            ei = _ebins[i];
            xi = x_ptr[i];
            integral_high = pow(ei, one_minus_PhoIndex)/one_minus_PhoIndex * exp(-xi)*hyp1f1(1.0, two_minus_PhoIndex, xi);
            fi = integral_high - integral_low;
            flux_ptr[i - 1] = (fi >= 0.0) ? fi : abs(fi);
            integral_low = integral_high;
        }

        integral_low = pow(ei, one_minus_PhoIndex) * expint_large_x_neg_v(PhoIndex, xi);
        for (py::ssize_t i = n + 1; i < _ebins.shape(0); i++) {
            ei = _ebins[i];
            xi = x_ptr[i];
            integral_high = pow(ei, one_minus_PhoIndex) * expint_large_x_neg_v(PhoIndex, xi);
            fi = integral_low - integral_high;
            flux_ptr[i - 1] = (fi >= 0.0) ? fi : abs(fi);
            integral_low = integral_high;
        }
    }

    return flux;
}

py::array_t<double> powerlaw(double PhoIndex, py::array_t<double> ebins) {
    auto _ebins = ebins.unchecked<1>();
    auto flux = py::array_t<double>(ebins.size() - 1);
    py::buffer_info flux_buf = flux.request();
    double *flux_ptr = static_cast<double *>(flux_buf.ptr);

    const double alpha = 1.0 - PhoIndex;
    if (alpha == 0.0) {
        double integral_low = log(_ebins[0]);
        double integral_high;
        for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
            integral_high = log(_ebins[i]);
            flux_ptr[i - 1] = integral_high - integral_low;
            integral_low = integral_high;
        }
    } else {
        const double inv_alpha = 1.0/alpha;
        double integral_low = inv_alpha*pow(_ebins[0], alpha);
        double integral_high;
        for (py::ssize_t i = 1; i < _ebins.shape(0); i++) {
            integral_high = inv_alpha*pow(_ebins[i], alpha);
            flux_ptr[i - 1] = integral_high - integral_low;
            integral_low = integral_high;
        }
    }

    return flux;
}

PYBIND11_MODULE(specfun, module) {
    module.doc() = "Spectral function library";
    module.def("bbody", &bbody, "bbody", py::arg("kT"), py::arg("ebins"));
    module.def("bbodyrad", &bbodyrad, "bbodyrad", py::arg("kT"), py::arg("ebins"));
    module.def("cutoffpl", &cutoffpl, "cutoffpl", py::arg("PhoIndex"), py::arg("HighECut"), py::arg("ebins"));
    module.def("powerlaw", &powerlaw, "powerlaw", py::arg("PhoIndex"), py::arg("ebins"));
}