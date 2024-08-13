#pragma once

#include "mer31_t.cuh"

template <typename T, const T* W>
class alignas(alignof(T) * 2) mer31_extension_t {

protected:
    union {
        struct {
            T x;
            T y;
        };
        T elems[2];
    };
    
public:
    HD inline constexpr mer31_extension_t(T a, T b) : x(a), y(b) {}
    HD inline constexpr mer31_extension_t(T a[2]) : x(a[0]), y(a[1]) {}
    HD inline mer31_extension_t(const mer31_extension_t& m) : x(m.x), y(m.y) {}
    
    HD inline constexpr T& operator[](size_t index) 
    {   return elems[index];    }
    HD inline constexpr const T& operator[](size_t index) const
    {   return elems[index];    }

    HD inline mer31_extension_t& operator+=(const mer31_extension_t b)
    {
        x += b.x;
        y += b.y;
        return *this;
    }
    friend HD inline mer31_extension_t operator+(mer31_extension_t a, const mer31_extension_t b)
    {   return a += b;   }

    HD inline mer31_extension_t operator-() const
    {   
        return mer31_extension_t(-x, -y);
    }
    HD inline mer31_extension_t& operator-=(const mer31_extension_t b)
    {
        *this += -b;
        return *this;
    }
    friend HD inline mer31_extension_t operator-(mer31_extension_t a, const mer31_extension_t b)
    {   return a -= b;   }


    HD inline mer31_extension_t& operator*=(const T b)
    {
        x *= b;
        y *= b;
        return *this;
    }
    friend HD inline mer31_extension_t operator*(mer31_extension_t a, const T b)
    {   return a *= b;   }

    HD inline mer31_extension_t& operator/=(const T b)
    {   
        *this *= b.reciprocal();   
        return *this;   
    }
    friend HD inline mer31_extension_t operator/(mer31_extension_t a, const T b)
    {   return a /= b;   }

    HD inline mer31_extension_t& operator*=(const mer31_extension_t b)
    {
        // if constexpr (*W == -T(1)) {
        //     x = (x * b.x) - (y * b.y);
        //     y = (x * b.y) + (y * b.x);
        // } else {
            x = (x * b.x) + (y * b.y * *W);
            y = (x * b.y) + (y * b.x);
        // }
        return *this;
    }
    friend HD inline mer31_extension_t operator*(mer31_extension_t a, const mer31_extension_t b)
    {   return a *= b;   }

    HD inline mer31_extension_t reciprocal() const
    {   
        // if constexpr (*W == -T(1)) {
        //     return mer31_extension_t(x, -y) / ((x * x) + (y * y));   
        // } else {
            return mer31_extension_t(x, -y) / ((x * x) - (y * y * *W));   
        // }
    }
    HD inline mer31_extension_t& operator/=(const mer31_extension_t b)
    {   return *this * b.reciprocal();   }
    friend HD inline mer31_extension_t operator/(mer31_extension_t a, const mer31_extension_t b)
    {   return a /= b;   }

    friend HD inline bool operator==(const mer31_extension_t& lhs, const mer31_extension_t& rhs) 
    {   return lhs.x == rhs.x && lhs.y == rhs.y;   }
    friend HD inline bool operator!=(const mer31_extension_t& lhs, const mer31_extension_t& rhs) 
    {   return !(lhs == rhs);   }
};


// Implementation of polynomial {x^2 + 1}
__device__ constexpr mer31_t W_complex = -mer31_t(1, mer31_t::by_value{});
using mer31_complex_t = mer31_extension_t<mer31_t, &W_complex>;

// // Implementation of polynomial {x^2 - (2,1)}
// __device__ constexpr mer31_complex_t W_ext128 = mer31_complex_t(
//     mer31_t(2, mer31_t::by_value{}), 
//     mer31_t(1, mer31_t::by_value{})
// );
// using mer31_ext128_t = mer31_extension_t<mer31_complex_t, &W_ext128>;

