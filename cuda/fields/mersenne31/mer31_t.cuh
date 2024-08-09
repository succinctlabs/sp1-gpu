#pragma once

#ifdef __CUDA_ARCH__
    #define HD __host__ __device__
    #define inline __forceinline__
#else
    #define HD
#endif

class alignas(32) mer31_t {

private:
    static const uint32_t M_31 = (1u<<31) - 1;

    template <typename T>
    HD static inline constexpr T reduce(T a) {
        return (a & M_31) + (a >> 31);
    }

    template <typename T>
    HD static inline constexpr T sub(T a) {
        return a - (a >= M_31) * M_31;
    }

    HD static inline uint64_t reduce64(uint64_t a) {
        return reduce(reduce(a));
    }

    HD static inline uint32_t mod(uint32_t a) {
        return sub(reduce(a));
    }

    HD static inline uint32_t mod(uint64_t a) {
        return static_cast<uint32_t>(sub(reduce64(a)));
    }

public:
    uint32_t val;

    struct by_value {};

    HD inline constexpr mer31_t(uint32_t v) : val(mod(v)) {}
    HD inline constexpr mer31_t(uint32_t v, by_value) : val(v) {}
    HD inline mer31_t(const mer31_t& m) : val(m.val) {}

    HD inline mer31_t& operator+=(const mer31_t b)
    {
        val = sub(val + b.val);
        return *this;
    }
    friend HD inline mer31_t operator+(mer31_t a, const mer31_t b)
    {   return a += b;   }

    HD inline mer31_t operator-() const
    {   
        return mer31_t(-(val != 0) & (M_31 - val), by_value{}); // do we need it?
    }
    HD inline mer31_t& operator-=(const mer31_t b)
    {
        *this += -b;
        return *this;
    }
    friend HD inline mer31_t operator-(mer31_t a, const mer31_t b)
    {   return a -= b;   }

    HD inline mer31_t& operator*=(const mer31_t b)
    {
        val = mod(static_cast<uint64_t>(val) * static_cast<uint64_t>(b.val));
        return *this;
    }
    friend HD inline mer31_t operator*(mer31_t a, const mer31_t b)
    {   return a *= b;   }

    HD inline mer31_t& operator^=(uint32_t p)
    {
        uint64_t base = static_cast<uint64_t>(val);
        uint64_t result = (p & 1u) * (base - 1) + 1;

        #pragma unroll
        while (p >>= 1) {
            base = reduce64(base * base);
            if (p & 1u)
                result = reduce64(result * base);
        }

        val = static_cast<uint32_t>(sub(result));
        return *this;
    }
    friend HD inline mer31_t operator^(mer31_t a, uint32_t p)
    {   return a ^= p;   }

    HD inline mer31_t reciprocal() const
    {   
        return *this ^= (M_31-2);   
    }
    HD inline bb31_t& operator/=(const bb31_t a)
    {   
        return *this *= a.reciprocal();   
    }
    friend HD inline bb31_t operator/(bb31_t a, bb31_t b)
    {   return a /= b;   }

    friend HD inline bool operator==(const bb31_t& lhs, const bb31_t& rhs) 
    {   return lhs.val == rhs.val;   }
    friend HD inline bool operator!=(const bb31_t& lhs, const bb31_t& rhs) 
    {   return !(lhs == rhs);   }
};