#pragma once

#include "mer31_t.cuh"

template <typename T>
class alignas(alignof(T) * 2) mer31_extension_t_base {

protected:
    union {
        struct {
            T x;
            T y;
        };
        T elems[2];
    };
    
public:
    HD inline constexpr mer31_extension_t_base(T a, T b) : x(a), y(b) {}
    HD inline constexpr mer31_extension_t_base(T a[2]) : x(a[0]), y(a[1]) {}
    HD inline mer31_extension_t_base(const mer31_extension_t_base& m) : x(m.x), y(m.y) {}
    
    HD inline constexpr T& operator[](size_t index) 
    {   return elems[index];    }
    HD inline constexpr const T& operator[](size_t index) const
    {   return elems[index];    }

    HD inline mer31_extension_t_base& operator+=(const mer31_extension_t_base b)
    {
        x += b.x;
        y += b.y;
        return *this;
    }
    friend HD inline mer31_extension_t_base operator+(mer31_extension_t_base a, const mer31_extension_t_base b)
    {   return a += b;   }

    HD inline mer31_extension_t_base operator-() const
    {   
        return mer31_extension_t_base(-x, -y);
    }
    HD inline mer31_extension_t_base& operator-=(const mer31_extension_t_base b)
    {
        *this += -b;
        return *this;
    }
    friend HD inline mer31_extension_t_base operator-(mer31_extension_t_base a, const mer31_extension_t_base b)
    {   return a -= b;   }


    HD inline mer31_extension_t_base& operator*=(const T b)
    {
        x *= b;
        y *= b;
        return *this;
    }
    friend HD inline mer31_extension_t_base operator*(mer31_extension_t_base a, const T b)
    {   return a *= b;   }

    HD inline mer31_extension_t_base& operator/=(const T b)
    {   
        *this *= b.reciprocal();   
        return *this;   
    }

    friend HD inline mer31_extension_t_base operator/(mer31_extension_t_base a, const T b)
    {   return a /= b;   }

    friend HD inline bool operator==(const mer31_extension_t_base& lhs, const mer31_extension_t_base& rhs) 
    {   return lhs.x == rhs.x && lhs.y == rhs.y;   }
    
    friend HD inline bool operator!=(const mer31_extension_t_base& lhs, const mer31_extension_t_base& rhs) 
    {   return !(lhs == rhs);   }
};

class mer31_complex_t : public mer31_extension_t_base<mer31_t> {
public:
    using mer31_extension_t_base::mer31_extension_t_base;

    HD inline mer31_complex_t(const mer31_extension_t_base<mer31_t>& base)
        : mer31_extension_t_base(base) {}

    HD inline mer31_complex_t(const mer31_t& x) 
        : mer31_extension_t_base(x, mer31_t(0, mer31_t::by_value{})) {}

    HD inline mer31_complex_t& operator*=(const mer31_complex_t b) 
    {
        x = (x * b.x) - (y * b.y);
        y = (x * b.y) + (y * b.x);

        return *this;
    }

    HD inline mer31_complex_t reciprocal() const 
    {
        return static_cast<mer31_extension_t_base<mer31_t>>(mer31_complex_t(x, -y)) /
               static_cast<mer31_t>((x * x) + (y * y));
    }

    HD inline mer31_complex_t& operator/=(const mer31_complex_t b)
    {
        return *this *= b.reciprocal();
    }

    friend HD inline mer31_complex_t operator/(mer31_complex_t a, mer31_complex_t b) 
    {
        return a /= b;
    }

    friend HD inline mer31_complex_t operator*(mer31_complex_t a, const mer31_complex_t b)
    {   
        return a *= b;   
    }
};

__device__ constexpr mer31_t W1_ext128 = mer31_t(2, mer31_t::by_value{});
__device__ constexpr mer31_t W2_ext128 = mer31_t(1, mer31_t::by_value{});

__device__ constexpr mer31_complex_t W_ext128 = mer31_complex_t(
        W1_ext128, 
        W2_ext128
    );

class mer31_ext128_t : public mer31_extension_t_base<mer31_complex_t> {
public:
    using mer31_extension_t_base::mer31_extension_t_base;

    HD inline mer31_ext128_t(const mer31_extension_t_base<mer31_complex_t>& base)
        : mer31_extension_t_base(base) {}

    HD inline mer31_ext128_t& operator*=(const mer31_ext128_t b) 
    {
        x = mer31_complex_t((x * b.x) + (y * b.y * W_ext128));
        y = mer31_complex_t((x * b.y) + (y * b.x));

        return *this;
    }

    HD inline mer31_ext128_t reciprocal() const 
    {
        return static_cast<mer31_extension_t_base<mer31_complex_t>>(mer31_ext128_t(x, y)) /
               static_cast<mer31_complex_t>((x * x) - (y * y * W_ext128));
    }

    HD inline mer31_ext128_t& operator/=(const mer31_ext128_t b)
    {
        return *this *= b.reciprocal();
    }

    friend HD inline mer31_ext128_t operator/(mer31_ext128_t a, mer31_ext128_t b) 
    {
        return a /= b;
    }

    friend HD inline mer31_ext128_t operator*(mer31_ext128_t a, const mer31_ext128_t b)
    {   
        return a *= b;   
    }
};