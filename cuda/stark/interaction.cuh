#pragma once

/*

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PairColDevice<F> {
    column_idx: usize,
    is_preprocessed: bool,
    weight: F,
}

#[derive(Debug)]
#[repr(C)]
pub struct DeviceInteractionsView<'a, F: Field> {
    pub values_ptr: &'a DeviceSlice<usize>,
    pub multiplicities_ptr: &'a DeviceSlice<usize>,
    pub values_col_weights_ptr: &'a DeviceSlice<usize>,

    pub values_col_weights: &'a DeviceSlice<PairColDevice<F>>,
    pub values_constants: &'a DeviceSlice<F>,

    pub mult_col_weights: &'a DeviceSlice<PairColDevice<F>>,
    pub mult_constants: &'a DeviceSlice<F>,

    pub arg_indices: &'a DeviceSlice<F>,
    pub is_send: &'a DeviceSlice<bool>,
    pub num_interactions: usize,
}

*/

template<typename F>struct PairCol {
    size_t column_idx;
    bool is_preprocessed;
    F weight;

public: 
    __device__ F get(Matrix<F> preprocessed, Matrix<F> main, size_t RowIdx) {
        if (is_preprocessed) {
            return preprocessed.values[column_idx * preprocessed.height + RowIdx] * weight;
        } else {
            return main.values[column_idx * main.height + RowIdx] * weight;
        }
    }
};


template<typename F> struct Interactions {
    size_t * values_ptr;
    size_t * multiplicities_ptr;
    size_t * values_col_weights_ptr;

    PairCol<F> * values_col_weights;
    F * values_constants;

    PairCol<F> * mult_col_weights;
    F * mult_constants;

    F * arg_indices;
    bool * is_sends;
    bool * is_globals;

    size_t num_global_interactions;
    size_t num_local_interactions;
};