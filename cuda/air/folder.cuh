#pragma once

template <typename Val, typename Challenge, size_t N>
struct ConstraintFolder {
   public:
    Val* prepLocal;
    Val* prepNext;
    Val* mainLocal;
    Val* mainNext;
    Val* publicValues;
    Challenge* permLocal;
    Challenge* permNext;
    Challenge* permChallenges;
    Challenge cumulativeSum;
    Val isFirstRow;
    Val isLastRow;
    Val isTransition;
    Challenge alpha;
    Challenge accumulator;

   public:
    __device__ ConstraintFolder() {}

    __device__ Challenge var(SymbolicFolderVar var) {
        switch (var.variant) {
            case SymbolicFolderVarType::Base:
                return bb31_extension_t(mainLocal[var.idx]);
            case SymbolicFolderVarType::Extension:
                return bb31_extension_t(mainNext[var.idx]);
            case SymbolicFolderVarType::PreprocessedLocal:
                return bb31_extension_t(prepLocal[var.idx]);
            case SymbolicFolderVarType::PreprocessedNext:
                return bb31_extension_t(prepNext[var.idx]);
            case SymbolicFolderVarType::MainLocal:
                return bb31_extension_t(mainLocal[var.idx]);
            case SymbolicFolderVarType::MainNext:
                return bb31_extension_t(mainNext[var.idx]);
            case SymbolicFolderVarType::PermutationLocal:
                return permLocal[var.idx];
            case SymbolicFolderVarType::PermutationNext:
                return permNext[var.idx];
            case SymbolicFolderVarType::PermutationChallenge:
                return permChallenges[var.idx];
            case SymbolicFolderVarType::CumulativeSum:
                return cumulativeSum;
            case SymbolicFolderVarType::PublicValue:
                return publicValues[var.idx];
            case SymbolicFolderVarType::IsFirstRow:
                return bb31_extension_t(isFirstRow);
            case SymbolicFolderVarType::IsLastRow:
                return bb31_extension_t(isLastRow);
            case SymbolicFolderVarType::IsTransition:
                return bb31_extension_t(isTransition);
            case SymbolicFolderVarType::Alpha:
                return alpha;
            case SymbolicFolderVarType::Accumulator:
                return accumulator;
            case SymbolicFolderVarType::Empty:
                return Challenge::zero();
        }
    }
};