#include "../fields/bb31_t.cuh"
#include "../hashes//poseidon2_bb31_16/poseidon2_bb31_16.cuh"

#include <stdio.h>

struct RowMajorMatrix {
    DeviceSlice<bb31_t> data;
    int width;
    int height;
};

class FieldMerkleTree {
   private:
    poseidon2_bb31_16::Hasher hasher;

   public:
    FieldMerkleTree(poseidon2_bb31_16::Hasher hasher) : hasher(hasher) {
        this->hasher = hasher;
    }

    __device__ void firstDigestLayer(
        DeviceSlice<RowMajorMatrix> tallestMatrices,
        DeviceSlice<bb31_t[poseidon2_bb31_16::DIGEST_WIDTH]> digests,
        size_t nRow) {
        poseidon2_bb31_16::HasherState state;
        for (int i = 0; i < tallestMatrices.length; i++) {
            DeviceSlice<bb31_t> row = tallestMatrices[i].data.slice(
                tallestMatrices[i].width * nRow, tallestMatrices[i].width);
            hasher.absorb(row, state);
        }
        hasher.finalize(state, digests[nRow]);
    }

    __device__ void compressAndInject(
        DeviceSlice<bb31_t[poseidon2_bb31_16::DIGEST_WIDTH]> prevLayer,
        DeviceSlice<RowMajorMatrix> matricesToInject,
        DeviceSlice<bb31_t[poseidon2_bb31_16::DIGEST_WIDTH]> nextDigests,
        size_t nRow) {
        if (matricesToInject.length == 0) {
            return;
        }

        size_t nextLen = matricesToInject[0].height;
        size_t nextLenPadded = prevLayer.length / 2;

        bb31_t defaultDigest[poseidon2_bb31_16::DIGEST_WIDTH] = {
            bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0)};

        bb31_t digest[poseidon2_bb31_16::DIGEST_WIDTH];
        hasher.compress(prevLayer[nRow * 2], prevLayer[nRow * 2 + 1], digest);

        if (nRow < nextLen) {
            bb31_t tallestDigest[poseidon2_bb31_16::DIGEST_WIDTH];
            poseidon2_bb31_16::HasherState state;
            for (int i = 0; i < matricesToInject.length; i++) {
                DeviceSlice<bb31_t> row = matricesToInject[i].data.slice(
                    matricesToInject[i].width * nRow,
                    matricesToInject[i].width);
                hasher.absorb(row, state);
            }
            hasher.finalize(state, tallestDigest);
            hasher.compress(digest, tallestDigest, nextDigests[nRow]);
        } else {
            hasher.compress(digest, defaultDigest, nextDigests[nRow]);
        }
    }
};

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