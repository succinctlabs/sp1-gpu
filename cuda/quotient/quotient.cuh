#pragma once

#include "./utils.cuh"

#include "../air/folder.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../matrix/matrix.cuh"

namespace quotient_kernels {
template <typename Val, typename Challenge, size_t MEMORY_SIZE>
__global__ void computeValues(Operation *evalProgram,
                              size_t evalProgramLen, Challenge cumulativeSum,
                              TwoAdicMultiplicativeCoset<Val> traceDomain,
                              TwoAdicMultiplicativeCoset<Val> quotientDomain,
                              Matrix<Val> preprocessedTraceOnQuotientDomain,
                              Matrix<Val> mainTraceOnQuotientDomain,
                              Matrix<Val> permutationTraceOnQuotientDomain,
                              Challenge *permChallenges, Challenge alpha,
                              Val *publicValues,
                              Val traceDomainGenerator,
                              Val* generatorPowers,
                            //   LagrangeSelectors<Val> selectors,
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
    folder.cumulativeSum = cumulativeSum;
    folder.isFirstRow = isFirstRow;
    folder.isLastRow = isLastRow;
    folder.isTransition = isTransition;
    folder.alpha = alpha;
    folder.accumulator = Challenge::zero();
    folder.quotientIdx = quotientIdx;
    folder.quotientSize = quotientSize;
    folder.nextStep = nextStep;

    Challenge expr[MEMORY_SIZE];
    for (size_t i = 0; i < MEMORY_SIZE; i++) {
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
    }

    folder.accumulator = expr[0];
    bb31_extension_t quotient_value = folder.accumulator * invZeroifier;

    #pragma unroll
        for (size_t k = 0; k < bb31_extension_t::D; k++) {
            quotientValues.values[k * quotientValues.height + quotientIdx] = quotient_value.value[k];
        }
}
}  // namespace quotient_kernels

namespace quotient_gpu {
extern "C" void computeValues(
    Operation *evalProgram, 
    size_t evalProgramLen,
    size_t memorySize,
    bb31_extension_t cumulativeSum,
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
    size_t numThreadsPerBlock) {
    if (memorySize <= 32) {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 32><<<numBlocks, numThreadsPerBlock>>>(  
            evalProgram, evalProgramLen, cumulativeSum,        
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues); 
    }
    else if (memorySize <= 64)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 64><<<numBlocks, numThreadsPerBlock>>>(  
            evalProgram, evalProgramLen, cumulativeSum,        
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 128)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 128><<<numBlocks, numThreadsPerBlock>>>(  
            evalProgram, evalProgramLen, cumulativeSum,        
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 256)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 256><<<numBlocks, numThreadsPerBlock>>>(  
            evalProgram, evalProgramLen, cumulativeSum,        
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 512)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 512><<<numBlocks, numThreadsPerBlock>>>(  
            evalProgram, evalProgramLen, cumulativeSum,        
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else if (memorySize <= 1024)
    {
        quotient_kernels::computeValues<bb31_t, bb31_extension_t, 1024><<<numBlocks, numThreadsPerBlock>>>(  
            evalProgram, evalProgramLen, cumulativeSum,        
            traceDomain, quotientDomain, preprocessedTraceOnQuotientDomain,  
            mainTraceOnQuotientDomain, permutationTraceOnQuotientDomain,     
            permChallenges, alpha, publicValues, traceDomainGenerator, generatorPowers, quotientValues);
    }
    else {
        assert(false);   
    }
    }
}  // namespace quotient_gpu
