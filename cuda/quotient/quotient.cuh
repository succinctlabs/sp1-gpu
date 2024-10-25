#pragma once

#include "./utils.cuh"

#include "../air/folder.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../matrix/matrix.cuh"

#define DEBUG_FLAG 0  // Set this to 0 or 1

#if DEBUG_FLAG == 1
    #define DEBUG(...) printf(__VA_ARGS__)
#else
    #define DEBUG(...)  // Do nothing
#endif

namespace quotient_kernels {
template <typename Val, typename Challenge, size_t MEMORY_SIZE>
__global__ void computeValues(Instruction *evalProgram,
                              size_t evalProgramLen, 
                              Challenge *evalConstants,
                              Challenge *cumulativeSums,
                              TwoAdicMultiplicativeCoset<Val> traceDomain,
                              TwoAdicMultiplicativeCoset<Val> quotientDomain,
                              Matrix<Val> preprocessedTraceOnQuotientDomain,
                              Matrix<Val> mainTraceOnQuotientDomain,
                              Matrix<Val> permutationTraceOnQuotientDomain,
                              Challenge *permChallenges, 
                              Challenge alpha,
                              Val *publicValues,
                              Val traceDomainGenerator,
                              Val* generatorPowers,
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


    Val generator = generatorPowers[1];
    Val blockGenerator = generator^(blockIdx.x * blockDim.x);

    Val point = blockGenerator * generatorPowers[threadIdx.x] * quotientDomain.shift; 

    LagrangeSelectorsAtPoint<Val> selectors = traceDomain.selectors_at_point(traceDomainGenerator, point);


    Val isFirstRow = selectors.is_first_row[quotientIdx];
    Val isLastRow = selectors.is_last_row[quotientIdx];
    Val isTransition = selectors.is_transition[quotientIdx];
    Val invZeroifier = selectors.inv_zeroifier[quotientIdx];

    ConstraintFolder<Val, Challenge, 2> folder =
        ConstraintFolder<Val, Challenge, 2>();
    folder.prep = preprocessedTraceOnQuotientDomain;
    folder.main = mainTraceOnQuotientDomain;
    folder.publicValues = publicValues;
    folder.perm = permutationTraceOnQuotientDomain;
    folder.permChallenges = permChallenges;
    folder.cumulativeSums = cumulativeSums;
    folder.isFirstRow = isFirstRow;
    folder.isLastRow = isLastRow;
    folder.isTransition = isTransition;
    folder.alpha = alpha;
    folder.accumulator = Challenge::zero();
    folder.quotientIdx = quotientIdx;
    folder.quotientSize = quotientSize;
    folder.nextStep = nextStep;


    Val expr_f[MEMORY_SIZE];
    for (size_t i = 0; i < MEMORY_SIZE; i++) {
        expr_f[i] = Val{0};
    }
    Challenge expr_ef[10];
    for (size_t i = 0; i < 10; i++) {
        expr_ef[i] = Challenge::zero();
    }

    for (size_t i = 0; i < evalProgramLen; i++) {
        Instruction instr = evalProgram[i]; 
        switch (instr.opcode) {
            case 0:
                DEBUG("EMPTY\n");
                break;

            case 1:
                DEBUG("FAssignC: %d <- %d\n", instr.a, instr.b);
                expr_f[instr.a] = bb31_t{instr.b};
                DEBUG("FAssignC Output: %d\n", expr_f[instr.a]);
                break;
            case 2:
                DEBUG("FAssignV: %d <- (%d, %d)\n", instr.a, instr.b_variant, instr.b);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b);
                break;
            case 3:
                DEBUG("FAssignE: %d <- %d\n", instr.a, instr.b);
                expr_f[instr.a] = expr_f[instr.b];
                break;

            case 4:
                DEBUG("FAddVC: %d <- %d + %d\n", instr.a, instr.b_variant, instr.b);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) + bb31_t{instr.c};
                break;
            case 5:
                DEBUG("FAddVV: %d <- (%d, %d) + (%d, %d)\n", instr.a, instr.b_variant, instr.b, instr.c_variant, instr.c);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) + folder.var_f(instr.c_variant, instr.c);
                break;
            case 6:
                DEBUG("FAddVE: %d <- (%d, %d) + %d\n", instr.a, instr.b_variant, instr.b, instr.c);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) + expr_f[instr.c];
                break;
            
            case 7:
                DEBUG("FAddEC: %d <- %d + %d\n", instr.a, instr.b_variant, instr.b);
                expr_f[instr.a] = expr_f[instr.b] + bb31_t{instr.c};
                break;
            case 8:
                DEBUG("FAddEV: %d <- %d + (%d, %d)\n", instr.a, instr.b, instr.c_variant, instr.c);
                expr_f[instr.a] = expr_f[instr.b] + folder.var_f(instr.c_variant, instr.c);
                break;
            case 9:
                DEBUG("FAddEE: %d <- %d + %d\n", instr.a, instr.b, instr.c);
                expr_f[instr.a] = expr_f[instr.b] + expr_f[instr.c];
                break;
            case 10:
                DEBUG("FAddAssignE: %d <- %d\n", instr.a, instr.b);
                expr_f[instr.a] += expr_f[instr.b];
                break;

            case 11:
                DEBUG("FSubVC: %d <- %d - %d\n", instr.a, instr.b_variant, instr.b);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) - bb31_t{instr.c};
                break;
            case 12:
                DEBUG("FSubVV: %d <- (%d, %d) - (%d, %d)\n", instr.a, instr.b_variant, instr.b, instr.c_variant, instr.c);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) - folder.var_f(instr.c_variant, instr.c);
                break;
            case 13:
                DEBUG("FSubVE: %d <- (%d, %d) - %d\n", instr.a, instr.b_variant, instr.b, instr.c);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) - expr_f[instr.c];
                break;
            
            case 14:
                DEBUG("FSubEC: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                expr_f[instr.a] = expr_f[instr.b] - bb31_t{instr.c};
                break;
            case 15:
                DEBUG("FSubEV: %d <- %d - (%d, %d)\n", instr.a, instr.b, instr.c_variant, instr.c);
                expr_f[instr.a] = expr_f[instr.b] - folder.var_f(instr.c_variant, instr.c);
                break;
            case 16:
                DEBUG("FSubEE: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                DEBUG("FSubEE Input: %d, %d\n", expr_f[instr.b], expr_f[instr.c]);
                expr_f[instr.a] = expr_f[instr.b] - expr_f[instr.c];
                DEBUG("FSubEE Output: %d\n", expr_f[instr.a]);
                break;
            case 17:
                DEBUG("FSubAssignE: %d <- %d\n", instr.a, instr.b);
                expr_f[instr.a] -= expr_f[instr.b];
                break;

            case 18:
                DEBUG("FMulVC: %d <- %d * %d\n", instr.a, instr.b_variant, instr.b);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) * bb31_t{instr.c};
                break;
            case 19:
                DEBUG("FMulVV: %d <- (%d, %d) * (%d, %d)\n", instr.a, instr.b_variant, instr.b, instr.c_variant, instr.c);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) * folder.var_f(instr.c_variant, instr.c);
                break;
            case 20:
                DEBUG("FMulVE: %d <- (%d, %d) * %d\n", instr.a, instr.b_variant, instr.b, instr.c);
                expr_f[instr.a] = folder.var_f(instr.b_variant, instr.b) * expr_f[instr.c];
                break;

            case 21:
                DEBUG("FMulEC: %d <- %d * %d\n", instr.a, instr.b_variant, instr.b);
                expr_f[instr.a] = expr_f[instr.b] * bb31_t{instr.c};
                break;
            case 22:
                DEBUG("FMulEV: %d <- %d * (%d, %d)\n", instr.a, instr.b, instr.c_variant, instr.c);
                expr_f[instr.a] = expr_f[instr.b] * folder.var_f(instr.c_variant, instr.c);
                break;
            case 23:
                DEBUG("FMulEE: %d <- %d * %d\n", instr.a, instr.b, instr.c);
                DEBUG("FMulEE Input: %d, %d\n", expr_f[instr.b], expr_f[instr.c]);
                expr_f[instr.a] = expr_f[instr.b] * expr_f[instr.c];
                DEBUG("FMulEE Output: %d\n", expr_f[instr.a]);
                break;
            case 24:
                DEBUG("FMulAssignE: %d <- %d\n", instr.a, instr.b);
                expr_f[instr.a] *= expr_f[instr.b];
                break;

            case 25:
                DEBUG("FNegE: %d <- -%d\n", instr.a, instr.b);
                expr_f[instr.a] = -expr_f[instr.b];
                break;

            case 26:
                DEBUG("EAssignC: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] = evalConstants[instr.b];
                break;
            case 27:
                DEBUG("EAssignV: %d <- (%d, %d)\n", instr.a, instr.b_variant, instr.b);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b);
                break;
            case 28:
                DEBUG("EAssignE: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] = expr_ef[instr.b];
                break;

            case 29:
                DEBUG("EAddVC: %d <- %d + %d\n", instr.a, instr.b_variant, instr.b);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) + bb31_t{instr.c};
                break;
            case 30:
                DEBUG("EAddVV: %d <- (%d, %d) + (%d, %d)\n", instr.a, instr.b_variant, instr.b, instr.c_variant, instr.c);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) + folder.var_ef(instr.c_variant, instr.c);
                break;
            case 31:
                DEBUG("EAddVE: %d <- (%d, %d) + %d\n", instr.a, instr.b_variant, instr.b, instr.c);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) + expr_ef[instr.c];
                break;
            
            case 32:
                DEBUG("EAddEC: %d <- %d + %d\n", instr.a, instr.b_variant, instr.b);
                expr_ef[instr.a] = expr_ef[instr.b] + bb31_t{instr.c};
                break;
            case 33:
                DEBUG("EAddEV: %d <- %d + (%d, %d)\n", instr.a, instr.b, instr.c_variant, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] + folder.var_ef(instr.c_variant, instr.c);
                break;
            case 34:
                DEBUG("EAddEE: %d <- %d + %d\n", instr.a, instr.b, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] + expr_ef[instr.c];
                break;
            case 35:
                DEBUG("EAddAssignE: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] += expr_ef[instr.b];
                break;

            case 36:
                DEBUG("ESubVC: %d <- %d - %d\n", instr.a, instr.b_variant, instr.b);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) - bb31_t{instr.c};
                break;
            case 37:
                DEBUG("ESubVV: %d <- (%d, %d) - (%d, %d)\n", instr.a, instr.b_variant, instr.b, instr.c_variant, instr.c);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) - folder.var_ef(instr.c_variant, instr.c);
                break;
            case 38:
                DEBUG("ESubVE: %d <- (%d, %d) - %d\n", instr.a, instr.b_variant, instr.b, instr.c);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) - expr_ef[instr.c];
                break;

            case 39:
                DEBUG("ESubEC: %d <- %d - %d\n", instr.a, instr.b_variant, instr.b);
                expr_ef[instr.a] = expr_ef[instr.b] - bb31_t{instr.c};
                break;
            case 40:
                DEBUG("ESubEV: %d <- %d - (%d, %d)\n", instr.a, instr.b, instr.c_variant, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] - folder.var_ef(instr.c_variant, instr.c);
                break;
            case 41:
                DEBUG("ESubEE: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] - expr_ef[instr.c];
                DEBUG("ESubEE Output: %d, %d, %d, %d\n", expr_ef[instr.a].value[0], expr_ef[instr.a].value[1], expr_ef[instr.a].value[2], expr_ef[instr.a].value[3]);
                break;
            case 42:
                DEBUG("ESubAssignE: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] -= expr_ef[instr.b];
                break;

            case 43:
                DEBUG("EMulVC: %d <- %d * %d\n", instr.a, instr.b_variant, instr.b);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) * bb31_t{instr.c};
                break;
            case 44:
                DEBUG("EMulVV: %d <- (%d, %d) * (%d, %d)\n", instr.a, instr.b_variant, instr.b, instr.c_variant, instr.c);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) * folder.var_ef(instr.c_variant, instr.c);
                break;
            case 45:
                DEBUG("EMulVE: %d <- (%d, %d) * %d\n", instr.a, instr.b_variant, instr.b, instr.c);
                expr_ef[instr.a] = folder.var_ef(instr.b_variant, instr.b) * expr_ef[instr.c];
                break;

            case 46:
                DEBUG("EMulEC: %d <- %d * %d\n", instr.a, instr.b_variant, instr.b);
                expr_ef[instr.a] = expr_ef[instr.b] * bb31_t{instr.c};
                break;
            case 47:
                DEBUG("EMulEV: %d <- %d * (%d, %d)\n", instr.a, instr.b, instr.c_variant, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] * folder.var_ef(instr.c_variant, instr.c);
                break;
            case 48:
                DEBUG("EMulEE: %d <- %d * %d\n", instr.a, instr.b, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] * expr_ef[instr.c];
                break;
            case 49:
                DEBUG("EMulAssignE: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] *= expr_ef[instr.b];
                break;

            case 50:
                DEBUG("ENegE: %d <- -%d\n", instr.a, instr.b);
                expr_ef[instr.a] = Challenge::zero() - expr_ef[instr.b];
                break;

            case 51:
                DEBUG("EFFromE: %d <- %d\n", instr.a, instr.b);
                Challenge result;
                result.value[0] = expr_f[instr.b];
                result.value[1] = Val{0};
                result.value[2] = Val{0};
                result.value[3] = Val{0};
                expr_ef[instr.a] = result;
                break;
            case 52:
                DEBUG("EFAddEE: %d <- %d + %d\n", instr.a, instr.b, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] + expr_f[instr.c];
                break;  
            case 53:
                DEBUG("EFAddAssignE: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] += expr_f[instr.b];
                break;
            case 54:
                DEBUG("EFSubEE: %d <- %d - %d\n", instr.a, instr.b, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] - expr_f[instr.c];
                break;
            case 55:
                DEBUG("EFSubAssignE: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] -= expr_f[instr.b];
                break;
            case 56:
                DEBUG("EFMulEE: %d <- %d * %d\n", instr.a, instr.b, instr.c);
                expr_ef[instr.a] = expr_ef[instr.b] * expr_f[instr.c];
                break;
            case 57:
                DEBUG("EFMulAssignE: %d <- %d\n", instr.a, instr.b);
                expr_ef[instr.a] *= expr_f[instr.b];
                break;
            case 58:
                DEBUG("EFAsBaseSlice: %d <- (%d, %d)\n", instr.a, instr.b_variant, instr.b);
                // TODO:
                break;

            case 59:
                DEBUG("FAssertZero: %d\n", instr.a);
                folder.accumulator *= folder.alpha;
                folder.accumulator += expr_f[instr.a];
                break;
            case 60:
                DEBUG("EAssertZero: %d\n", instr.a);
                folder.accumulator *= folder.alpha;
                folder.accumulator += expr_ef[instr.a];
                break;
        } 
    }

    bb31_extension_t quotient_value = folder.accumulator * invZeroifier;

    #pragma unroll
    for (size_t k = 0; k < bb31_extension_t::D; k++) {
        quotientValues.values[k * quotientValues.height + quotientIdx] = quotient_value.value[k];
    }
}
}  // namespace quotient_kernels

namespace quotient_gpu {
extern "C" void computeValues(
    Instruction *evalProgram, 
    size_t evalProgramLen,
    bb31_extension_t *evalConstants,
    size_t memorySize,
    bb31_extension_t *cumulativeSums,
    TwoAdicMultiplicativeCoset<bb31_t> traceDomain,
    TwoAdicMultiplicativeCoset<bb31_t> quotientDomain,
    Matrix<bb31_t> preprocessedTraceOnQuotientDomain,
    Matrix<bb31_t> mainTraceOnQuotientDomain,
    Matrix<bb31_t> permutationTraceOnQuotientDomain,
    bb31_extension_t *permChallenges, 
    bb31_extension_t alpha,
    bb31_t *publicValues, 
    bb31_t traceDomainGenerator, 
    bb31_t* generatorPowers,
    Matrix<bb31_t> quotientValues, 
    size_t numBlocks,
    size_t numThreadsPerBlock,
    cudaStream_t stream) {

    if (memorySize <= 32) {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 32><<<numBlocks, numThreadsPerBlock, 0, stream>>>(  
            evalProgram, evalProgramLen, evalConstants, cumulativeSums,
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues); 
    }
    else if (memorySize <= 64)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 64><<<numBlocks, numThreadsPerBlock, 0, stream>>>(  
            evalProgram, evalProgramLen, evalConstants, cumulativeSums,
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 128)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 128><<<numBlocks, numThreadsPerBlock, 0, stream>>>(  
            evalProgram, evalProgramLen, evalConstants, cumulativeSums,
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 256)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 256><<<numBlocks, numThreadsPerBlock, 0, stream>>>(  
            evalProgram, evalProgramLen, evalConstants, cumulativeSums,
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 512)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 512><<<numBlocks, numThreadsPerBlock, 0, stream>>>(  
            evalProgram, evalProgramLen, evalConstants, cumulativeSums,
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 1024)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 1024><<<numBlocks, numThreadsPerBlock, 0, stream>>>(  
            evalProgram, evalProgramLen, evalConstants, cumulativeSums,
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else {
        assert(false);   
    }
    }
}  // namespace quotient_gpu
