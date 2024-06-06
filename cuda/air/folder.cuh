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
};