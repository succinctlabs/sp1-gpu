#pragma once

template <typename Val, typename Challenge, size_t N>
struct ConstraintFolder {
   public:
    Matrix<Val> prep;
    Matrix<Val> main;
    Val* publicValues;
    Matrix<Val> perm;
    Challenge* permChallenges;
    Challenge* cumulativeSums;
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

    // __device__ Challenge var(SymbolicFolderVar var) {
    //     switch (var.variant) {
    //         case SymbolicFolderVarType::Base:
    //             return bb31_extension_t(main.values[var.idx * main.height + 
    //             (quotientIdx % quotientSize)]);
    //         case SymbolicFolderVarType::Extension:
    //             return bb31_extension_t(main.values[var.idx * main.height + 
    //             ((quotientIdx + nextStep) % quotientSize)]);
    //         case SymbolicFolderVarType::PreprocessedLocal:
    //             return bb31_extension_t(prep.values[var.idx * prep.height + 
    //             (quotientIdx % quotientSize)]);
    //         case SymbolicFolderVarType::PreprocessedNext:
    //             return bb31_extension_t(prep.values[var.idx * prep.height + 
    //             ((quotientIdx + nextStep) % quotientSize)]);
    //         case SymbolicFolderVarType::MainLocal:
    //             return bb31_extension_t(main.values[var.idx * main.height + 
    //             (quotientIdx % quotientSize)]);
    //         case SymbolicFolderVarType::MainNext:
    //             return bb31_extension_t(main.values[var.idx * main.height + 
    //             ((quotientIdx + nextStep) % quotientSize)]);
    //         case SymbolicFolderVarType::PermutationLocal:
    //             {
    //             Challenge result;
    //             for (size_t k = 0 ; k < Challenge::D; k++)
    //                 result.value[k] = perm.values[(var.idx * Challenge::D + k) * perm.height + 
    //                    (quotientIdx % quotientSize)];
    //             return result;
    //             }
    //         case SymbolicFolderVarType::PermutationNext:
    //             {
    //             Challenge result;
    //             for (size_t k = 0 ; k < Challenge::D; k++)
    //                 result.value[k] = perm.values[(var.idx * Challenge::D + k) * perm.height + 
    //                 ((quotientIdx + nextStep) % quotientSize)];
    //             return result;
    //             }
    //         case SymbolicFolderVarType::PermutationChallenge:
    //             return permChallenges[var.idx];
    //         case SymbolicFolderVarType::CumulativeSum:
    //             return cumulativeSums[var.idx];
    //         case SymbolicFolderVarType::PublicValue:
    //             return publicValues[var.idx];
    //         case SymbolicFolderVarType::IsFirstRow:
    //             return bb31_extension_t(isFirstRow);
    //         case SymbolicFolderVarType::IsLastRow:
    //             return bb31_extension_t(isLastRow);
    //         case SymbolicFolderVarType::IsTransition:
    //             return bb31_extension_t(isTransition);
    //         case SymbolicFolderVarType::Alpha:
    //             return alpha;
    //         case SymbolicFolderVarType::Accumulator:
    //             return accumulator;
    //         case SymbolicFolderVarType::Empty:
    //             return Challenge::zero();
    //     }
    // }
};