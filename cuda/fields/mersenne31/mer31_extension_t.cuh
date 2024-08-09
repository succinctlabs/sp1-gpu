#pragma once

#include "mer31_t.cuh"

template <typename T>
class mer31_extension_t {

private:
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
        x = -x;
        y = -y;
        return *this;
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
    {   return *this * b.reciprocal();   }
    friend HD inline mer31_extension_t operator/(mer31_extension_t a, const T b)
    {   return a /= b;   }

    HD virtual inline mer31_extension_t& operator*=(const mer31_extension_t b) = 0;
    friend HD inline mer31_extension_t operator*(mer31_extension_t a, const mer31_extension_t b)
    {   return a *= b;   }

    HD virtual inline mer31_extension_t reciprocal() const = 0;
    HD inline mer31_extension_t& operator/=(const mer31_extension_t b)
    {   return *this * b.reciprocal();   }
    friend HD inline mer31_extension_t operator/(mer31_extension_t a, const mer31_extension_t b)
    {   return a /= b;   }

    friend HD inline bool operator==(const bb31_extension_t& lhs, const bb31_extension_t& rhs) 
    {   return lhs.x == rhs.x && lhs.y == rhs.y;   }
    friend HD inline bool operator!=(const bb31_extension_t& lhs, const bb31_extension_t& rhs) 
    {   return !(lhs == rhs);   }
};


// Implementation of polynomial {x^2 + 1}
class mer31_complex_t : public mer31_extension_t<mer31_t> {

public:
    using mer31_extension_t::mer31_extension_t;

    HD inline mer31_complex_t reciprocal() const
    {   
        return mer31_complex_t(x, -y) / ((x * x) + (y * y));   
    }

    HD inline mer31_complex_t& operator*=(const mer31_complex_t b)
    {
        x = (x * b.x) - (y * b.y);
        y = (x * b.y) + (y * b.x);
        return *this;
    }
};

// Implementation of polynomial {x^2 - x - 2}
class mer31_ext128_t : public mer31_extension_t<mer31_complex_t> {

public:
    using mer31_extension_t::mer31_extension_t;

    HD inline mer31_ext128_t reciprocal() const
    {   
        return mer31_complex_t(x + y, -y) / ((x * (x + y)) -  (y * (y + y)));  
    }

    HD inline mer31_ext128_t& operator*=(const mer31_ext128_t b)
    {
        x = (x * b.x) + (y * (b.y + b.y));
        y = (x * b.y) + (y * (b.x + b.y));
        return *this;
    }
};

