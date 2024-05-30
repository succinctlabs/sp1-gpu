
#pragma once

#include "folder.cuh"

template <typename T>
class FibonacciAir {
   public:
    __device__ eval(ConstraintFolder<T> builder) {
        T a_local = builder.main_local()[0];
        T b_local = builder.main_local()[1];

        T a_next = builder.main_next()[0];
        T b_next = builder.main_next()[1];

        // Assert the initial conditions.
        T a_initial = builder.public_inputs()[0];
        T b_initial = builder.public_inputs()[1];

        builder

            // Assert the transition conditions b' = a + b, a' = b.
            builder.assert_zero(builder.is_transition() * (a_next - b_local));
        builder.assert_zero(builder.is_transition() *
                            (b_next - (a_local + b_local)));
    }
};
