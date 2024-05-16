// #include "../fields/bb31_t.cuh"
// #include "../poseidon2/poseidon2.cuh"

// #include <stdio.h>

// struct RowMajorMatrix {
//     bb31_t *data;
//     int width;
//     int height;
// };

// class FieldMerkleTree {
//    private:
//     Poseidon2 hasher;

//    public:
//     FieldMerkleTree(Poseidon2 hasher) : hasher(hasher) {
//         this->hasher = hasher;
//     }

//     __device__ void firstDigestLayer(
//         DeviceSlice<RowMajorMatrix> tallestMatrices,
//         DeviceSlice<bb31_t> concatenatedRow, bb31_t digest[DIGEST_WIDTH]) {
//         int rowIdx = 0;
//         for (int i = 0; i < tallestMatrices.length; i++) {
//             for (int j = 0; j < tallestMatrices[i].width; j++) {
//                 concatenatedRow[rowIdx] = tallestMatrices[i].data[j];
//                 rowIdx++;
//             }
//         }
//         hasher.hash(concatenatedRow, digest);
//     }
// };

// __global__ void vectorFirstDigestLayer(
//     FieldMerkleTree tree, DeviceSlice<RowMajorMatrix> tallestMatrices,
//     bb31_t *concatenatedRows, int concatenatedRowWidth,
//     DeviceSlice<bb31_t[DIGEST_WIDTH]> digests) {
//     int height = tallestMatrices[0].height;
//     int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//     if (idx >= height) {
//         return;
//     }

//     DeviceSlice<bb31_t> concatenatedRow = DeviceSlice(
//         concatenatedRows + idx * concatenatedRowWidth, concatenatedRowWidth);
//     bb31_t *digest = digests[idx];
//     tree.firstDigestLayer(tallestMatrices, concatenatedRow, digest);
// }