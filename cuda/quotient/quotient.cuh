

#include "../air/codegen/byte.cuh"
#include "../air/folder.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../utils/matrix.cuh"

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

namespace quotient {
template <typename Air, typename Val, typename Challenge>
__global__ void quotientValues(Air air, Challenge cumulativeSum,
                               TwoAdicMultiplicativeCoset<Val> traceDomain,
                               TwoAdicMultiplicativeCoset<Val> quotientDomain,
                               Matrix<Val> preprocessedTraceOnQuotientDomain,
                               Matrix<Val> mainTraceOnQuotientDomain,
                               Matrix<Val> permutationTraceOnQuotientDomain,
                               Challenge *permChallenges, Challenge alpha,
                               Val *publicValues,
                               LagrangeSelectors<Val> selectors,
                               Challenge *quotientValues) {
    size_t quotientSize = quotientDomain.size();
    size_t prepWidth = preprocessedTraceOnQuotientDomain.width;
    size_t mainWidth = mainTraceOnQuotientDomain.width;
    size_t permWidth = permutationTraceOnQuotientDomain.width;
    size_t qdb = quotientDomain.log_n - traceDomain.log_n;
    size_t nextStep = 1 << qdb;

    size_t quotientIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (quotientIdx >= quotientSize) {
        return;
    }

    Val prepLocal[Air::PREP_WIDTH + 1];
    Val prepNext[Air::PREP_WIDTH + 1];
    for (size_t i = 0; i < Air::PREP_WIDTH; i++) {
        prepLocal[i] =
            preprocessedTraceOnQuotientDomain
                .values[i * preprocessedTraceOnQuotientDomain.height +
                        (quotientIdx % quotientSize)];
        prepNext[i] = preprocessedTraceOnQuotientDomain
                          .values[i * preprocessedTraceOnQuotientDomain.height +
                                  ((quotientIdx + nextStep) % quotientSize)];
    }

    Val mainLocal[Air::MAIN_WIDTH + 1];
    Val mainNext[Air::MAIN_WIDTH + 1];
    for (size_t i = 0; i < Air::MAIN_WIDTH; i++) {
        mainLocal[i] = mainTraceOnQuotientDomain
                           .values[i * mainTraceOnQuotientDomain.height +
                                   (quotientIdx % quotientSize)];
        mainNext[i] = mainTraceOnQuotientDomain
                          .values[i * mainTraceOnQuotientDomain.height +
                                  ((quotientIdx + nextStep) % quotientSize)];
    }

    Challenge permLocal[Air::PERM_WIDTH + 1];
    Challenge permNext[Air::PERM_WIDTH + 1];
    for (size_t i = 0; i < Air::PERM_WIDTH; i++) {
        for (size_t j = 0; j < 4; j++) {
            permLocal[i].value[j] =
                permutationTraceOnQuotientDomain
                    .values[(i * 4 + j) *
                                permutationTraceOnQuotientDomain.height +
                            (quotientIdx % quotientSize)];
            permNext[i].value[j] =
                permutationTraceOnQuotientDomain
                    .values[(i * 4 + j) *
                                permutationTraceOnQuotientDomain.height +
                            ((quotientIdx + nextStep) % quotientSize)];
            if (quotientIdx == 65536) {
                printf("permLocal[%d].value[%d] = %d\n", i, j,
                       permLocal[i].value[j]);
                printf("permNext[%d].value[%d] = %d\n", i, j,
                       permNext[i].value[j]);
            }
        }
    }

    Val isFirstRow = selectors.is_first_row[quotientIdx];
    Val isLastRow = selectors.is_last_row[quotientIdx];
    Val isTransition = selectors.is_transition[quotientIdx];
    Val invZeroifier = selectors.inv_zeroifier[quotientIdx];

    ConstraintFolder<Val, Challenge, 2> folder =
        ConstraintFolder<Val, Challenge, 2>();
    folder.prepLocal = prepLocal;
    folder.prepNext = prepNext;
    folder.mainLocal = mainLocal;
    folder.mainNext = mainNext;
    folder.permLocal = permLocal;
    folder.permNext = permNext;
    folder.permChallenges = permChallenges;
    folder.cumulativeSum = cumulativeSum;
    folder.isFirstRow = isFirstRow;
    folder.isLastRow = isLastRow;
    folder.isTransition = isTransition;
    folder.alpha = alpha;
    folder.accumulator = Challenge::zero();

    air.eval(&folder);
    quotientValues[quotientIdx] = folder.accumulator * invZeroifier;
}
}  // namespace quotient

extern "C" void quotient_values(
    bb31_extension_t cumulativeSum,
    TwoAdicMultiplicativeCoset<bb31_t> traceDomain,
    TwoAdicMultiplicativeCoset<bb31_t> quotientDomain,
    Matrix<bb31_t> preprocessedTraceOnQuotientDomain,
    Matrix<bb31_t> mainTraceOnQuotientDomain,
    Matrix<bb31_t> permutationTraceOnQuotientDomain,
    bb31_extension_t *permChallenges, bb31_extension_t alpha,
    bb31_t *publicValues, LagrangeSelectors<bb31_t> selectors,
    bb31_extension_t *quotientValues) {
    ByteAir air;
    printf("quotientValues: %p\n", quotientValues);
    quotient::quotientValues<<<4096, 32>>>(
        air, cumulativeSum, traceDomain, quotientDomain,
        preprocessedTraceOnQuotientDomain, mainTraceOnQuotientDomain,
        permutationTraceOnQuotientDomain, permChallenges, alpha, publicValues,
        selectors, quotientValues);
    cudaDeviceSynchronize();
}