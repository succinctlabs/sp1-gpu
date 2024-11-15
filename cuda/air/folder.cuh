#pragma once

template <typename Val, typename Challenge, typename GlobalSum, size_t N>
struct ConstraintFolder {
   public:
    Matrix<Val> prep;
    Matrix<Val> main;
    Val* publicValues;
    Matrix<Val> perm;
    Challenge* permChallenges;
    Challenge localCumulativeSum;
    GlobalSum globalCumulativeSum;
    Val isFirstRow;
    Val isLastRow;
    Val isTransition;
    Challenge alpha;
    Challenge accumulator;
    size_t nextStep;
    size_t quotientIdx;
    size_t quotientSize;

   public:
    __device__ ConstraintFolder() {}

    __inline__ __device__ Val var_f(unsigned char variant, unsigned int idx) {
        switch (variant) {
            case 0:
                return Val{0};
            case 1:
                return Val{idx};
            case 2:
                return prep.values[idx * prep.height + (quotientIdx % quotientSize)];
            case 3:
                return prep.values[idx * prep.height + ((quotientIdx + nextStep) % quotientSize)];
            case 4:
                return main.values[idx * main.height + (quotientIdx % quotientSize)];
            case 5:
                return main.values[idx * main.height + ((quotientIdx + nextStep) % quotientSize)];
            case 6:
                return isFirstRow;
            case 7:
                return isLastRow;
            case 8:
                return isTransition;
            case 9:
                return publicValues[idx];
            case 10:
                if (idx < 7) {
                    return globalCumulativeSum.point.x.value[idx];
                }
                else {
                    return globalCumulativeSum.point.y.value[idx - 7];
                }
        }
    }

    __inline__ __device__ Challenge var_ef(unsigned char variant, unsigned int idx) {
        switch (variant) {
            case 0:
                return Challenge::zero();
            case 1:
                {
                Challenge result;
                for (size_t k = 0 ; k < Challenge::D; k++)
                    result.value[k] = perm.values[(idx * Challenge::D + k) * perm.height + 
                        (quotientIdx % quotientSize)];
                return result;
                }
            case 2:
                {
                Challenge result;
                for (size_t k = 0 ; k < Challenge::D; k++)
                    result.value[k] = perm.values[(idx * Challenge::D + k) * perm.height + 
                    ((quotientIdx + nextStep) % quotientSize)];
                return result;
                }
            case 3:
                return permChallenges[idx];
            case 4:
                return localCumulativeSum;
        }
    }
};