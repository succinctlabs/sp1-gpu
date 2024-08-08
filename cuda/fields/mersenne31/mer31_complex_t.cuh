#pragma once

#ifdef __CUDA_ARCH__
    #define HD __host__ __device__
    #define inline __forceinline__
#else
    #define HD
#endif

#include "mer31_t.cuh"


// Implementation of x+yi (for {x^2 + 1} polinomial)
class mer31_complex_t {

private:

public:
    mer31_t x;
    mer31_t y;

    HD inline constexpr mer31_complex_t(mer31_t a, mer31_t b) : x(a), y(b) {}
    HD inline constexpr mer31_complex_t(mer31_t a[2]) : x(a[0]), y(a[1]) {}
    HD inline mer31_complex_t(const mer31_complex_t& m) : x(m.x), y(m.y) {}
    
    HD inline mer31_complex_t& operator+=(const mer31_complex_t b)
    {
        x += b.x;
        y += b.y;
        return *this;
    }
    friend HD inline mer31_complex_t operator+(mer31_complex_t a, const mer31_complex_t b)
    {   return a += b;   }

    HD inline mer31_complex_t operator-() const
    {   
        x = -x;
        y = -y;
        return *this;
    }
    HD inline mer31_complex_t& operator-=(const mer31_complex_t b)
    {
        *this += -b;
        return *this;
    }
    friend HD inline mer31_complex_t operator-(mer31_complex_t a, const mer31_complex_t b)
    {   return a -= b;   }


    HD inline mer31_complex_t& operator*=(const mer31_t b)
    {
        x *= b;
        y *= b;
        return *this;
    }
    friend HD inline mer31_complex_t operator*(mer31_complex_t a, const mer31_t b)
    {   return a *= b;   }

    HD inline mer31_complex_t& operator/=(const mer31_t b)
    {   return *this * b.reciprocal();   }
    friend HD inline mer31_complex_t operator/(mer31_complex_t a, const mer31_t b)
    {   return a /= b;   }

    HD inline mer31_t reciprocal() const
    {   return mer31_complex_t(x, -y) / ((x * x) + (y * y));   }


    HD inline mer31_complex_t& operator*=(const mer31_complex_t b)
    {
        x = x * b.x - y * b.y;
        y = x * b.y + y * b.x;
        return *this;
    }
    friend HD inline mer31_complex_t operator*(mer31_complex_t a, const mer31_complex_t b)
    {   return a *= b;   }

    HD inline mer31_complex_t& operator/=(const mer31_complex_t b)
    {   return *this * b.reciprocal();   }
    friend HD inline mer31_complex_t operator/(mer31_complex_t a, const mer31_complex_t b)
    {   return a /= b;   }

    friend HD inline bool operator==(const bb31_extension_t& lhs, const bb31_extension_t& rhs) 
    {   return lhs.x == rhs.x && lhs.y == rhs.y;   }
    friend HD inline bool operator!=(const bb31_extension_t& lhs, const bb31_extension_t& rhs) 
    {   return !(lhs == rhs);   }
};