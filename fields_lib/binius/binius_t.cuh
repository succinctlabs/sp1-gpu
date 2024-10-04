#pragma once


/*
bin8 is a class for level 3 extension of binius Tower x_k^2 = x_k * x_[k-1] + 1
The reduction rules are:
x0^2 = x0 + 1;
x1^2 = x0x1 + 1;
x2^2 = x1x2 + 1;
These rules are applied to 8bit representation of binius Tower, where
8 bits are {x0x1x2, x1x2, x0x2, x2, x0x1, x1, x0, 1} in polynimial view

Lookup table is calculated for reducing math & conditions in multiplication
Each 8 bits of this table is a result of multiplication 1 bit to another
with applying reduction rules. For example, 64th element is
x0^2 * x1^2 * x2^2 = x1x2 + x2 + x1 + x0 + 1 -> 01010111

It is packed as 64 bit value for reducing IO. Each value is a result of 
multiplying 1 bit to 8bit one by one in descending order
*/

#define SIZE_8 8
#define MASK_8 0xFF

static const uint64_t h_bit8_mul_constants[SIZE_8] = {
    0x8040201008040201LL, // 10000000,01000000,00100000,00010000,00001000,00000100,00000010,00000001
    0xC08030200C080302LL, // 11000000,10000000,00110000,00100000,00001100,00001000,00000011,00000010
    0xE09080400E090804LL, // 11100000,10010000,10000000,01000000,00001110,00001001,00001000,00000100
    0x70E0C080070E0C08LL, // 01110000,11100000,11000000,10000000,00000111,00001110,00001100,00001000
    0xE894824180402010LL, // 11101000,10010100,10000010,01000001,10000000,01000000,00100000,00010000
    0x7CE8C382C0803020LL, // 01111100,11101000,11000011,10000010,11000000,10000000,00110000,00100000
    0xFEA9E894E0908040LL, // 11111110,10101001,11101000,10010100,11100000,10010000,10000000,01000000
    0x57FE7CE870E0C080LL  // 01010111,11111110,01111100,11101000,01110000,11100000,11000000,10000000
};

__constant__ uint64_t d_bit8_mul_constants[SIZE_8];

struct constant_init {
    constant_init() {
        cudaMemcpyToSymbol(d_bit8_mul_constants, h_bit8_mul_constants, sizeof(h_bit8_mul_constants));
    }
};
static constant_init initializer;


#ifdef __CUDA_ARCH__
    #define HD __host__ __device__
    #define inline __forceinline__
    #define BIT8_CONST d_bit8_mul_constants
#else
    #define HD
    #define BIT8_CONST h_bit8_mul_constants
#endif


class bin8_t {
public:
    uint8_t val;

    HD inline constexpr bin8_t(const uint8_t v) : val(v) {}
    HD inline constexpr bin8_t(const bin8_t& v) : val(v.val) {}

    HD inline bin8_t& operator+=(const bin8_t b)
    {
        val ^= b.val;
        return *this;
    }
    friend HD inline bin8_t operator+(bin8_t a, const bin8_t b)
    {   return a += b;   }

    HD inline bin8_t& operator*=(const bin8_t b)
    {
        uint8_t product = (val & 0x01) ? val : 0;
        uint64_t mul_const = 0;
        int8_t b_bitmask[SIZE_8];
#pragma unroll
        for (int i = 0; i < SIZE_8; i++) {
            b_bitmask[i] = -(static_cast<int8_t>((b.val >> i) & 1));    // 0 or 0xFF
        }

#pragma unroll
        for (int i = 1; i < SIZE_8; i++) {
            if (val & (1 << i)) {
                mul_const = BIT8_CONST[i];
#pragma unroll
                for (int j = 0; j < SIZE_8; j++) {
                    product ^= (b_bitmask[j] & static_cast<uint8_t>(mul_const & MASK_8));
                    mul_const >>= SIZE_8;
                }
            }    
        }
        val = product;
        return *this;
    }
    friend HD inline bin8_t operator*(bin8_t a, const bin8_t b)
    {   return a *= b;   }
};

template<typename T>
class binius_t;

template<typename T>
HD T binius_reduction();

/*
    Base binius class uses a pair of {a, b}: a + b * x_k
    The reduction rule is x_k^2 = x_k * x_[k-1] + 1 -> {1, x_[k-1]}
    Where k starts from 3 because first extensions are in bin8_t.
*/
template<typename T>
class binius_t {
private:
    union {
        struct {
            T a;
            T b;
        };
        T elems[2];
    };

public:
    HD inline constexpr binius_t(T x, T y): a(x),    b(y) {}
    HD inline constexpr binius_t(T x[2])  : a(x[0]), b(x[1]) {}
    HD inline binius_t(const binius_t& m) : a(m.a),  b(m.b) {}
    HD inline constexpr binius_t(size_t bits): 
        a(T(bits)), b(T(bits >> (sizeof(T) * 8))) {}

    HD inline constexpr T& operator[](size_t index) 
    {   return elems[index];    }
    HD inline constexpr const T& operator[](size_t index) const
    {   return elems[index];    }

    HD inline binius_t& operator+=(const binius_t other)
    {
        a += other.a;
        b += other.b;
        return *this;
    }
    friend HD inline binius_t operator+(binius_t x, const binius_t y)
    {   return x += y;   }

    HD inline binius_t& operator-=(const binius_t other)
    {   return *this += other;   }
    friend HD inline binius_t operator-(binius_t x, const binius_t y)
    {   return x -= y;   }

    HD inline binius_t& operator*=(const T c)
    {
        a *= c;
        b *= c;
        return *this;
    }
    friend HD inline binius_t operator*(binius_t x, const T c)
    {   return x *= c;   }
    friend HD inline binius_t operator*(const T c, binius_t x)
    {   return x *= c;   }

    HD inline binius_t& operator*=(const binius_t other) 
    {
        T tmp = a * other.b + b * (other.a + other.b * binius_reduction<T>());
        a = a * other.a + b * other.b;
        b = tmp;
        return *this;
    }
    friend HD inline binius_t operator*(binius_t x, const binius_t y)
    {   return x *= y;   }
};

template<>
HD constexpr bin8_t binius_reduction() {
    return bin8_t(0x10);    // x2
}
using bin16_t = binius_t<bin8_t>;

template<>
HD constexpr bin16_t binius_reduction() {
    return bin16_t(0x0100);  // x3
}
using bin32_t = binius_t<bin16_t>;

template<>
HD constexpr bin32_t binius_reduction() {
    return bin32_t(0x00010000);  // x4
}
using bin64_t = binius_t<bin32_t>;

template<>
HD constexpr bin64_t binius_reduction() {
    return bin64_t(0x0000000100000000);   // x5
}
using bin128_t = binius_t<bin64_t>;