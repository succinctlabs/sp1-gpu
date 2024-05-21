#pragma once

#include "bb31_t.cuh"

class bb31_extension_t {
   public:
    static constexpr size_t D = 4;
    static constexpr bb31_t W = bb31_t(11);

    bb31_t values[D];

    inline bb31_extension_t() {}

    inline bb31_extension_t(bb31_t values[4]) {
        for (size_t i = 0; i < D; i++) {
            this->values[i] = values[i];
        }
    }

    inline bb31_extension_t(bb31_t value) {
        values[0] = value;
        for (size_t i = 1; i < D; i++) {
            values[i] = bb31_t(0);
        }
    }

    static inline const bb31_extension_t zero() {
        bb31_t values[D] = {bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0)};
        return bb31_extension_t(values);
    }

    static inline const bb31_extension_t one() {
        bb31_t values[D] = {bb31_t(1), bb31_t(0), bb31_t(0), bb31_t(0)};
        return bb31_extension_t(values);
    }

    inline bb31_extension_t& operator+=(const bb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            values[i] += b.values[i];
        }
        return *this;
    }

    friend inline bb31_extension_t operator+(bb31_extension_t a,
                                             const bb31_extension_t b) {
        return a += b;
    }

    inline bb31_extension_t& operator-=(const bb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            values[i] -= b.values[i];
        }
        return *this;
    }

    friend inline bb31_extension_t operator-(bb31_extension_t a,
                                             const bb31_extension_t b) {
        return a -= b;
    }

    inline bb31_extension_t& operator*=(const bb31_extension_t b) {
        bb31_t product[4];
        for (size_t i = 0; i < D; i++) {
            for (size_t j = 0; j < D; j++) {
                if (i + j >= 4) {
                    product[i + j - 4] += values[i] * b.values[i] * bb31_t(11);
                } else {
                    product[i + j] += values[i] * b.values[j];
                }
            }
        }

        for (size_t i = 0; i < D; i++) {
            values[i] = product[i];
        }

        return *this;
    }

    inline bb31_extension_t& operator*=(const bb31_t b) {
        for (size_t i = 0; i < D; i++) {
            values[i] *= b;
        }
        return *this;
    }

    friend inline bb31_extension_t operator*(bb31_extension_t a,
                                             const bb31_extension_t b) {
        return a *= b;
    }

    friend inline bb31_extension_t operator*(bb31_extension_t a,
                                             const bb31_t b) {
        return a *= b;
    }

    inline bb31_extension_t frobenius() {
        bb31_t z0 = bb31_t(1728404513);
        bb31_t z = z0;
        bb31_extension_t result;
        for (size_t i = 0; i < D; i++) {
            result.values[i] = values[i] * z;
            z *= z0;
        }
        return result;
    }

    inline bb31_extension_t frobeniusInverse() {
        bb31_extension_t f = one();
        for (size_t i = 1; i < D; i++) {
            f = (f * *this).frobenius();
        }

        bb31_extension_t a = *this;
        bb31_extension_t b = f;
        bb31_t g = bb31_t(0);
        for (size_t i = 1; i < D; i++) {
            g += a.values[i] * b.values[4 - i];
        }
        g *= bb31_t(11);
        g += a.values[0] * b.values[0];
        return f * g.reciprocal();
    }

    inline bb31_extension_t reciprocal() const {
        bool isZero = true;
        for (size_t i = 0; i < D; i++) {
            if (values[i] != 0) {
                isZero = false;
                break;
            }
        }

        if (isZero) {
            return zero();
        }

        return reciprocal();
    }
};