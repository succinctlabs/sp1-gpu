#pragma once

/**  A constraint folder for calculating the constraint polynomial on device. */
template <typename Val, typename Challenge, size_t N>
struct ConstraintFolder {
   public:
    Val* preprocessed_local;
    Val* preprocessed_next;
    Val* main_local;
    Val* main_next;

    Val* public_values;

    Val* permutation_local;
    Val* permutation_next;

    Challenge permutation_challenges[N];

    Challenge cumulative_sum;

    Challenge is_first_row;
    Challenge is_last_row;
    Challenge is_transition;
    Challenge alpha;
    Challenge accumulator;

   public:
    ConstraintFolder(Val* prep_local, Val* prep_next, Val* main_local,
                     Val* main_next, Val* perm_local, Val* perm_next,
                     Challenge alpha, Challenge is_first_row,
                     Challenge is_last_row, Challenge is_transition,
                     Challenge[N] permutation_challenges,
                     Challenge cumulative_sum, Challenge accumulator) {
        this->preprocessed_local = prep_local;
        this->preprocessed_next = prep_next;
        this->main_local = main_local;
        this->main_next = main_next;
        this->permutation_local = perm_local;
        this->permutation_next = perm_next;
        this->alpha = alpha;
        this->is_first_row = is_first_row;
        this->is_last_row = is_last_row;
        this->is_transition = is_transition;
        this->permutation_challenges = permutation_challenges;
        this->cumulative_sum = cumulative_sum;
        this->accumulator = accumulator;
    }

    Challenge is_first_row() { return is_first_row; }
    Challenge is_last_row() { return is_last_row; }
    Challenge is_transition() { return is_transition; }
    Challenge accumulator() { return accumulator; }

    Val* preprocessed_local() { return preprocessed_local; }
    Val* preprocessed_next() { return preprocessed_next; }
    Val* main_local() { return main_local; }
    Val* main_next() { return main_next; }
    Challenge* permutation_local() { return permutation_local; }
    Challenge* permutation_next() { return permutation_next; }

    void assert_zero(Val val) {
        x = Challenge::from_base(val);
        accumulator *= alpha;
        accumulator += x;
    }

    void assert_eq(Val lhs, Val rhs) { assert_zero(lhs - rhs); }

    void assert_zero_ext(Challenge c) {
        accumulator *= alpha;
        accumulator += c;
    }

    void assert_eq_ext(Challenge lhs, Challenge rhs) {
        assert_zero_ext(lhs - rhs);
    }
};