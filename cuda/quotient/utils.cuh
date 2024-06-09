template <typename Val>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;
    Val shift;

    __device__ size_t size() const { return 1 << log_n; }
};

template <typename Val>
struct LagrangeSelectors {
    Val *is_first_row;
    Val *is_last_row;
    Val *is_transition;
    Val *inv_zeroifier;
};