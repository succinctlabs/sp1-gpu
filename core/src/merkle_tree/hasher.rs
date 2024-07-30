use p3_field::Field;

use crate::matrix::MatrixViewDevice;

pub trait FieldMerkleTreeHasher<F: Field> {
    type Digest: Copy;

    /// # Safety
    ///
    /// TODO
    unsafe fn first_digest_layer(
        &self,
        tallest_matrices: *const MatrixViewDevice<F>,
        n_tallest_matrices: usize,
        digests: *mut Self::Digest,
        max_height: usize,
    );

    /// # Safety
    ///
    /// TODO
    unsafe fn compress_and_inject(
        &self,
        prev_layer: *const Self::Digest,
        matrices_to_inject: *const MatrixViewDevice<F>,
        n_matrices_to_inject: usize,
        next_digests: *mut Self::Digest,
        layer_len: usize,
    );
}
