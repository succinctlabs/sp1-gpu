pub mod merkle_tree_gpu {
    use p3_baby_bear::BabyBear;

    use crate::{
        device::slice::DeviceSliceRaw,
        poseidon2::poseidon2_bb31_16_gpu::{DIGEST_WIDTH, WIDTH},
    };

    #[derive(Clone, Copy)]
    #[repr(C)]
    pub struct RowMajorMatrixRaw {
        pub data: DeviceSliceRaw<BabyBear>,
        pub width: usize,
        pub height: usize,
    }

    #[allow(unused_attributes)]
    #[link_name = "merkle_tree_gpu"]
    extern "C" {
        #[link_name = "firstDigestLayer"]
        pub fn first_digest_layer(
            tallest_matrices: DeviceSliceRaw<RowMajorMatrixRaw>,
            digests: DeviceSliceRaw<[BabyBear; DIGEST_WIDTH]>,
            external_rc: DeviceSliceRaw<[BabyBear; WIDTH]>,
            internal_rc: DeviceSliceRaw<BabyBear>,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        #[link_name = "compressAndInject"]
        pub fn compress_and_inject(
            prev_layer: DeviceSliceRaw<[BabyBear; WIDTH]>,
            matrices_to_inject: DeviceSliceRaw<RowMajorMatrixRaw>,
            next_digests: DeviceSliceRaw<[BabyBear; DIGEST_WIDTH]>,
            external_rc: DeviceSliceRaw<[BabyBear; WIDTH]>,
            internal_rc: DeviceSliceRaw<BabyBear>,
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_first_digest_layer() {
        println!("test");
    }
}
