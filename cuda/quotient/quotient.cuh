#include "./utils.cuh"

#include "../air/codegen/codegen.cuh"
#include "../air/folder.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../matrix/matrix.cuh"

#define QUOTIENT(CHIP_TYPE)                                                  \
    {                                                                        \
        CHIP_TYPE chipInstance;                                              \
        quotient_kernels::computeValues<<<numBlocks, numThreadsPerBlock>>>(  \
            chipInstance, evalProgram, evalProgramLen, cumulativeSum,        \
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  \
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     \
            permChallenges, alpha, publicValues, selectors, quotientValues); \
    }

namespace quotient_kernels {
template <typename Air, typename Val, typename Challenge>
__global__ void computeValues(Air air, Operation *evalProgram,
                              size_t evalProgramLen, Challenge cumulativeSum,
                              TwoAdicMultiplicativeCoset<Val> traceDomain,
                              TwoAdicMultiplicativeCoset<Val> quotientDomain,
                              Matrix<Val> preprocessedTraceOnQuotientDomain,
                              Matrix<Val> mainTraceOnQuotientDomain,
                              Matrix<Val> permutationTraceOnQuotientDomain,
                              Challenge *permChallenges, Challenge alpha,
                              Val *publicValues,
                              LagrangeSelectors<Val> selectors,
                              Matrix<Val> quotientValues) {
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
    folder.publicValues = publicValues;
    folder.permLocal = permLocal;
    folder.permNext = permNext;
    folder.permChallenges = permChallenges;
    folder.cumulativeSum = cumulativeSum;
    folder.isFirstRow = isFirstRow;
    folder.isLastRow = isLastRow;
    folder.isTransition = isTransition;
    folder.alpha = alpha;
    folder.accumulator = Challenge::zero();

    Challenge expr[512];
    for (size_t i = 0; i < 512; i++) {
        expr[i] = Challenge::zero();
    }
    for (size_t i = 0; i < evalProgramLen; i++) {
        Operation op = evalProgram[i];
        switch (op.variant) {
            case OperationType::AssignF:
                expr[op.a.value] = bb31_extension_t(op.b_f);
                break;
            case OperationType::AssignEF:
                expr[op.a.value] = op.b_ef;
                break;
            case OperationType::AssignV:
                expr[op.a.value] = folder.var(op.b_var);
                break;
            case OperationType::AssignE:
                expr[op.a.value] = expr[op.b_expr.value];
                break;

            case OperationType::AddVF:
                expr[op.a.value] = folder.var(op.b_var) + op.c_f;
                break;
            case OperationType::AddVV:
                expr[op.a.value] = folder.var(op.b_var) + folder.var(op.c_var);
                break;
            case OperationType::AddVE:
                expr[op.a.value] = folder.var(op.b_var) + expr[op.c_expr.value];
                break;
            case OperationType::AddEF:
                expr[op.a.value] = expr[op.b_expr.value] + op.c_f;
                break;
            case OperationType::AddEV:
                expr[op.a.value] = expr[op.b_expr.value] + folder.var(op.c_var);
                break;
            case OperationType::AddEE:
                expr[op.a.value] =
                    expr[op.b_expr.value] + expr[op.c_expr.value];
                break;
            case OperationType::AddAssignE:
                expr[op.a.value] += expr[op.b_expr.value];
                break;

            case OperationType::SubVF:
                expr[op.a.value] = folder.var(op.b_var) - op.c_f;
                break;
            case OperationType::SubVV:
                expr[op.a.value] = folder.var(op.b_var) - folder.var(op.c_var);
                break;
            case OperationType::SubVE:
                expr[op.a.value] = folder.var(op.b_var) - expr[op.c_expr.value];
                break;
            case OperationType::SubEF:
                expr[op.a.value] = expr[op.b_expr.value] - op.c_f;
                break;
            case OperationType::SubEV:
                expr[op.a.value] = expr[op.b_expr.value] - folder.var(op.c_var);
                break;
            case OperationType::SubEE:
                expr[op.a.value] =
                    expr[op.b_expr.value] - expr[op.c_expr.value];
                break;
            case OperationType::SubAssignE:
                expr[op.a.value] = expr[op.a.value] - expr[op.b_expr.value];
                break;

            case OperationType::MulVF:
                expr[op.a.value] = folder.var(op.b_var) * op.c_f;
                break;
            case OperationType::MulVV:
                expr[op.a.value] = folder.var(op.b_var) * folder.var(op.c_var);
                break;
            case OperationType::MulVE:
                expr[op.a.value] = folder.var(op.b_var) * expr[op.c_expr.value];
                break;
            case OperationType::MulEF:
                expr[op.a.value] = expr[op.b_expr.value] * op.c_f;
                break;
            case OperationType::MulEV:
                expr[op.a.value] = expr[op.b_expr.value] * folder.var(op.c_var);
                break;
            case OperationType::MulEE:
                expr[op.a.value] =
                    expr[op.b_expr.value] * expr[op.c_expr.value];
                break;
            case OperationType::MulAssignE:
                expr[op.a.value] *= expr[op.b_expr.value];
                break;
            case OperationType::MulAssignEF:
                expr[op.a.value] *= op.b_ef;
                break;

            case OperationType::NegE:
                expr[op.a.value] =
                    (bb31_extension_t::zero() - bb31_extension_t::one()) *
                    expr[op.b_expr.value];
                break;
        }

        folder.accumulator = expr[0];
        bb31_extension_t quotient_value = folder.accumulator * invZeroifier;

        #pragma unroll
            for (size_t k = 0; k < bb31_extension_t::D; k++) {
                quotientValues.values[k * quotientValues.height + quotientIdx] = quotient_value.value[k];
            }

        // quotientValues[quotientIdx] = folder.accumulator * invZeroifier;
    }
}
}  // namespace quotient_kernels

namespace quotient_gpu {
extern "C" void computeValues(
    size_t chipId, Operation *evalProgram, size_t evalProgramLen,
    bb31_extension_t cumulativeSum,
    TwoAdicMultiplicativeCoset<bb31_t> traceDomain,
    TwoAdicMultiplicativeCoset<bb31_t> quotientDomain,
    Matrix<bb31_t> preprocessedTraceOnQuotientDomain,
    Matrix<bb31_t> mainTraceOnQuotientDomain,
    Matrix<bb31_t> permutationTraceOnQuotientDomain,
    bb31_extension_t *permChallenges, bb31_extension_t alpha,
    bb31_t *publicValues, LagrangeSelectors<bb31_t> selectors,
    Matrix<bb31_t> quotientValues, size_t numBlocks,
    size_t numThreadsPerBlock) {
    switch (chipId) {
        case 0:
            QUOTIENT(CPUAir);
            break;
        case 1:
            QUOTIENT(ProgramAir);
            break;
        case 2:
            QUOTIENT(ShaExtendAir);
            break;
        case 3:
            QUOTIENT(ShaCompressAir);
            break;
        case 4:
            QUOTIENT(EdAddAssignAir);
            break;
        case 5:
            QUOTIENT(EdDecompressAir);
            break;
        case 6:
            QUOTIENT(Secp256k1DecompressAir);
            break;
        case 7:
            QUOTIENT(Secp256k1AddAssignAir);
            break;
        case 8:
            QUOTIENT(Secp256k1DoubleAssignAir);
            break;
        case 9:
            QUOTIENT(KeccakPermuteAir);
            break;
        case 10:
            QUOTIENT(Bn254AddAssignAir);
            break;
        case 11:
            QUOTIENT(Bn254DoubleAssignAir);
            break;
        case 12:
            QUOTIENT(Bls12381AddAssignAir);
            break;
        case 13:
            QUOTIENT(Bls12381DoubleAssignAir);
            break;
        case 14:
            QUOTIENT(Uint256MulModAir);
            break;
        case 15:
            QUOTIENT(Bls12381DecompressAir);
            break;
        case 16:
            QUOTIENT(DivRemAir);
            break;
        case 17:
            QUOTIENT(AddSubAir);
            break;
        case 18:
            QUOTIENT(BitwiseAir);
            break;
        case 19:
            QUOTIENT(MulAir);
            break;
        case 20:
            QUOTIENT(ShiftRightAir);
            break;
        case 21:
            QUOTIENT(ShiftLeftAir);
            break;
        case 22:
            QUOTIENT(LtAir);
            break;
        case 23:
            QUOTIENT(MemoryInitAir);
            break;
        case 24:
            QUOTIENT(MemoryFinalizeAir);
            break;
        case 25:
            QUOTIENT(MemoryProgramAir);
            break;
        case 26:
            QUOTIENT(ByteAir);
            break;
    } 
}
}  // namespace quotient_gpu
