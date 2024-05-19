use crate::device::buffer::DeviceBuffer;
use crate::device::buffer::ToDevice;
use rand::Rng;

#[derive(Debug)]
#[repr(C)]
pub struct RowMajorMatrixDevice<T: Copy> {
    pub data: DeviceBuffer<T>,
    pub height: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RowMajorMatrixViewDevice<T> {
    pub data: *const T,
    pub width: usize,
    pub height: usize,
}

impl<T: Copy> RowMajorMatrixDevice<T> {
    pub fn rand(width: usize, height: usize) -> Self
    where
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let data = (0..width * height).map(|_| rng.gen()).collect::<Vec<_>>();
        RowMajorMatrixDevice {
            data: data.to_device(),
            height,
        }
    }

    pub fn view(&self) -> RowMajorMatrixViewDevice<T> {
        RowMajorMatrixViewDevice {
            data: self.data.as_slice().as_ptr(),
            width: self.data.len() / self.height,
            height: self.height,
        }
    }
}

pub mod merkle_tree_gpu {
    use p3_baby_bear::BabyBear;

    use crate::merkle_tree::RowMajorMatrixDevice;
    use crate::poseidon2::poseidon2_bb31_16_gpu::{DIGEST_WIDTH, WIDTH};

    #[allow(unused_attributes)]
    #[link_name = "merkle_tree_gpu"]
    extern "C" {
        #[link_name = "firstDigestLayer"]
        pub fn first_digest_layer(
            tallest_matrices: *const RowMajorMatrixDevice<BabyBear>,
            n_tallest_matrices: usize,
            digests: *mut [BabyBear; DIGEST_WIDTH],
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        #[link_name = "compressAndInject"]
        pub fn compress_and_inject(
            prev_layer: *const [BabyBear; WIDTH],
            n_prev_layer: usize,
            matrices_to_inject: *const RowMajorMatrixDevice<BabyBear>,
            n_matrices_to_inject: usize,
            next_digests: *mut [BabyBear; DIGEST_WIDTH],
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
