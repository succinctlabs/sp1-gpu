#include "../fields/bb31_t.cuh"
#include "../poseidon2/poseidon2.cuh"

#include <stdio.h>

struct RowMajorMatrixDevice {
    bb31_t *data;
    int width;
    int height;
};

class FieldMerkleTreeDevice {
   private:
    Poseidon2Device hasher;

   public:
    FieldMerkleTreeDevice(Poseidon2Device hasher) : hasher(hasher) {
        this->hasher = hasher;
    }

    __device__ void hashConcatenatedRows(RowMajorMatrixDevice *matrices,
                                         int nMatrices, bb31_t *concatenatedRow,
                                         int concatenatedRowWidth,
                                         bb31_t *digest) {
        int rowIdx = 0;
        for (int i = 0; i < nMatrices; i++) {
            for (int j = 0; j < matrices[i].width; j++) {
                concatenatedRow[rowIdx] = matrices[i].data[j];
                rowIdx++;
            }
        }
        hasher.hash(concatenatedRow, concatenatedRowWidth, digest);
    }
};

__global__ void vectorHashConcatenatedRows(
    FieldMerkleTreeDevice tree, RowMajorMatrixDevice *matrices, int nMatrices,
    bb31_t *concatenatedRows, int concatenatedRowWidth, bb31_t *digests) {
    int height = matrices[0].height;
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= height) {
        return;
    }

    bb31_t *concatenatedRow = concatenatedRows + idx * concatenatedRowWidth;
    bb31_t *digest = digests + idx * DIGEST_WIDTH;
    tree.hashConcatenatedRows(matrices, nMatrices, concatenatedRow,
                              concatenatedRowWidth, digest);
}