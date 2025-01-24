#pragma once

// #include "prelude.hpp"
// #include "utils.hpp"
// #include "bb31_septic_extension_t.hpp"

using namespace sp1_core_machine_sys;
using namespace sp1_recursion_core_sys::poseidon2;

__device__ void populate_global_interaction(GlobalInteractionOperation<bb31_t>* cols, const GlobalInteractionEvent* event) {
    // Initialize `m_trial` to the first 7 elements of the message.


    #pragma unroll(1)
    for(uint32_t offset = 0 ; offset < 256 ; offset++) { 
        bb31_t m_trial[POSEIDON2_WIDTH];
        {
            m_trial[0] = bb31_t::from_canonical_u32(event->message[0]) + bb31_t::from_canonical_u32(uint32_t(event->kind) << 16);
            m_trial[1] = bb31_t::from_canonical_u32(event->message[1]);
            m_trial[2] = bb31_t::from_canonical_u32(event->message[2]);
            m_trial[3] = bb31_t::from_canonical_u32(event->message[3]);
            m_trial[4] = bb31_t::from_canonical_u32(event->message[4]);
            m_trial[5] = bb31_t::from_canonical_u32(event->message[5]);
            m_trial[6] = bb31_t::from_canonical_u32(event->message[6]);
            m_trial[7] = bb31_t::from_canonical_u32(offset);
            m_trial[8] = bb31_t::zero();
            m_trial[9] = bb31_t::zero();
            m_trial[10] = bb31_t::zero();
            m_trial[11] = bb31_t::zero();
            m_trial[12] = bb31_t::zero();
            m_trial[13] = bb31_t::zero();
            m_trial[14] = bb31_t::zero();
            m_trial[15] = bb31_t::zero();
        }
        // Set the 8th element of `x_trial` to the offset.

        // Compute the poseidon2 hash of `m_trial` to compute `m_hash`.
        bb31_t m_hash[POSEIDON2_WIDTH];
        poseidon2::BabyBearHasher hasher;
        hasher.permute(m_trial, m_hash);

        // Convert the hash to a septic extension element.
        bb31_septic_extension_t x_trial = bb31_septic_extension_t::zero();
        for (uint32_t i = 0 ; i < 7 ; i++) {
            x_trial.value[i] = m_hash[i];
        }

        bb31_septic_extension_t y_sq = x_trial.curve_formula();
        bb31_t y_sq_pow_r = y_sq.pow_r();
        bb31_t is_square = y_sq_pow_r ^ 1006632960;
        if(is_square == bb31_t::one()) {
            bb31_septic_extension_t y = y_sq.sqrt(y_sq_pow_r);
            if (y.is_exception()) {
                continue;
            }
            if (y.is_receive() != event->is_receive) {
                y = bb31_septic_extension_t::zero() - y;
            }
            for(uint32_t idx = 0 ; idx < 8 ; idx++ ) {
                cols->offset_bits[idx] = bb31_t::from_canonical_u32((offset >> idx) & 1);
            }
            for(uintptr_t i = 0 ; i < 7 ; i++) {
                cols->x_coordinate._0[i] = x_trial.value[i];
                cols->y_coordinate._0[i] = y.value[i];
            }
            uint32_t range_check_value;
            if (event->is_receive) {
                range_check_value = y.value[6].as_canonical_u32() - 1;
            } else {
                range_check_value = y.value[6].as_canonical_u32() - (bb31_t::MOD + 1) / 2;
            }
            bb31_t top_4_bits = bb31_t::zero();
            for(uint32_t idx = 0 ; idx < 30 ; idx++) {
                cols->y6_bit_decomp[idx] = bb31_t::from_canonical_u32((range_check_value >> idx) & 1);
                if (idx >= 26) {
                    top_4_bits += cols->y6_bit_decomp[idx];
                }
            }
            top_4_bits -= bb31_t::from_canonical_u32(4);
            cols->range_check_witness = top_4_bits.reciprocal();

            bb31_t* input_row = reinterpret_cast<bb31_t*>(&cols->permutation);
            sp1_recursion_core_sys::poseidon2_wide::event_to_row(m_trial, input_row, 0, 1, true);

            return;
        }
        // x_start += bb31_t::from_canonical_u32(1 << 16);
    }
    assert(false);
}

__device__ void populate_global_interaction_dummy(GlobalInteractionOperation<bb31_t>* cols) {
    bb31_t m_trial[POSEIDON2_WIDTH];
    {
        m_trial[0] = bb31_t::zero();
        m_trial[1] = bb31_t::zero();
        m_trial[2] = bb31_t::zero();
        m_trial[3] = bb31_t::zero();
        m_trial[4] = bb31_t::zero();
        m_trial[5] = bb31_t::zero();
        m_trial[6] = bb31_t::zero();
        m_trial[7] = bb31_t::zero();
        m_trial[8] = bb31_t::zero();
        m_trial[9] = bb31_t::zero();
        m_trial[10] = bb31_t::zero();
        m_trial[11] = bb31_t::zero();
        m_trial[12] = bb31_t::zero();
        m_trial[13] = bb31_t::zero();
        m_trial[14] = bb31_t::zero();
        m_trial[15] = bb31_t::zero();
    } 

    bb31_t* input_row = reinterpret_cast<bb31_t*>(&cols->permutation);
    sp1_recursion_core_sys::poseidon2_wide::event_to_row(m_trial, input_row, 0, 1, true);
}


    // template<class bb31_t, class bb31_septic_extension_t>
    // __SP1_HOSTDEV__ void event_to_row(const MemoryLocalEvent* event, SingleMemoryLocal<bb31_t>* cols) {
    //     // populate_memory<bb31_t, bb31_septic_extension_t>(&cols->initial_global_interaction_cols, &event->initial_mem_access, event->addr, true);
    //     // populate_memory<bb31_t, bb31_septic_extension_t>(&cols->final_global_interaction_cols, &event->final_mem_access, event->addr, false);
    //     cols->addr = bb31_t::from_canonical_u32(event->addr);
        
    //     cols->initial_shard = bb31_t::from_canonical_u32(event->initial_mem_access.shard);
    //     cols->initial_clk = bb31_t::from_canonical_u32(event->initial_mem_access.timestamp);
    //     write_word_from_u32_v2<bb31_t>(cols->initial_value, event->initial_mem_access.value);
        
    //     cols->final_shard = bb31_t::from_canonical_u32(event->final_mem_access.shard);
    //     cols->final_clk = bb31_t::from_canonical_u32(event->final_mem_access.timestamp);
    //     write_word_from_u32_v2<bb31_t>(cols->final_value, event->final_mem_access.value);

    //     cols->is_real = bb31_t::one();
    // }
