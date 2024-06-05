#include "../air/folder.cuh"
#include "../utils/matrix.cuh"

template <typename Val>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;
    Val shift;

    size_t size() const { return 1 << log_n; }
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
__global__ void quotient_values(Challenge cumulativeSum,
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

    Val prepLocal[Air::PREP_WIDTH];
    Val prepNext[Air::PREP_WIDTH];
    for (size_t i = 0; i < Air::PREP_WIDTH; i++) {
        prepLocal[i] = preprocessedTraceOnQuotientDomain
                           .values[quotientIdx * Air::PREP_WIDTH + i];
        prepNext[i] =
            preprocessedTraceOnQuotientDomain
                .values[(quotientIdx + nextStep) * Air::PREP_WIDTH + i];
    }

    Val mainLocal[Air::MAIN_WIDTH];
    Val mainNext[Air::MAIN_WIDTH];
    for (size_t i = 0; i < Air::MAIN_WIDTH; i++) {
        mainLocal[i] =
            mainTraceOnQuotientDomain.values[quotientIdx * Air::MAIN_WIDTH + i];
        mainNext[i] =
            mainTraceOnQuotientDomain
                .values[(quotientIdx + nextStep) * Air::MAIN_WIDTH + i];
    }

    Challenge permLocal[Air::PERM_WIDTH];
    Challenge permNext[Air::PERM_WIDTH];
    for (size_t i = 0; i < Air::PERM_WIDTH; i++) {
        permLocal[i] = permutationTraceOnQuotientDomain
                           .values[quotientIdx * Air::PERM_WIDTH + i];
        permNext[i] =
            permutationTraceOnQuotientDomain
                .values[(quotientIdx + nextStep) * Air::PERM_WIDTH + i];
    }

    Val isFirstRow = selectors.is_first_row[quotientIdx];
    Val isLastRow = selectors.is_last_row[quotientIdx];
    Val isTransition = selectors.is_transition[quotientIdx];
    Val invZeroifier = selectors.inv_zeroifier[quotientIdx];
    Challenge accumulator = Challenge::zero();
    ConstraintFolder<Val, Challenge, 2> folder =
        ConstraintFolder(prepLocal, prepNext, mainLocal, mainNext, permLocal,
                         permNext, alpha, isFirstRow, isLastRow, isTransition,
                         permChallenges, cumulativeSum, accumulator);
    Air::eval(folder);
    quotientValues[quotientIdx] = folder.accumulator * invZeroifier;
}

}  // namespace quotient