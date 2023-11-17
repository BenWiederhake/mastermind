#!/usr/bin/env python3

from collections import Counter, defaultdict
from math import log


#COLORS = "RYGBOL"
#SLOTS = 4
COLORS = "RYGBOLSC"
SLOTS = 5


def gen_combinations(slots=SLOTS):
    assert slots >= 0
    if slots == 0:
        yield ()
        return
    for combination in gen_combinations(slots - 1):
        for color in COLORS:
            yield (*combination, color)


def evaluate(possibility, guess):
    nomatch_possibility = Counter()
    nomatch_guess = Counter()
    correct = 0
    assert len(possibility) == len(guess)
    for color_poss, color_guess in zip(possibility, guess):
        if color_poss == color_guess:
            correct += 1
        else:
            nomatch_possibility[color_poss] += 1
            nomatch_guess[color_guess] += 1
    badloc = 0
    for color, count_poss in nomatch_possibility.items():
        count_guess = nomatch_guess[color]
        count_common = min(count_poss, count_guess)
        badloc += count_common
    assert correct + badloc <= len(possibility), (possibility, guess, correct, badloc, len(possibility))
    return (correct, badloc)


def test_evaluate():
    data = [
        ("RYGB", "RYGB", (4, 0)),
        ("RYGB", "RYGO", (3, 0)),
        ("RYGB", "YGBR", (0, 4)),
        ("RYGB", "YGLO", (0, 2)),
        ("RYGB", "OOOO", (0, 0)),
        ("OORG", "OOBR", (2, 1)),
    ]
    for code_a, code_b, expect in data:
        actual1 = evaluate(code_a, code_b)
        actual2 = evaluate(code_b, code_a)
        assert actual1 == actual2 == expect, (code_a, code_b, expect, actual1, actual2)


def apply_restriction(possibilities, guess, expect_correct, expect_badlocs):
    remaining_possibilities = []
    for possibility in possibilities:
        actual_correct, actual_badlocs = evaluate(possibility, guess)
        if (actual_correct, actual_badlocs) == (expect_correct, expect_badlocs):
            remaining_possibilities.append(possibility)
    return remaining_possibilities


def quality_by_max_bucket(results_to_possibility_count):
    # Bigger is worse
    return max(results_to_possibility_count.values())


def quality_by_entropy(results_to_possibility_count):
    n = sum(results_to_possibility_count.values())
    neg_entropy = 0
    for k in results_to_possibility_count.values():
        p = k / n
        neg_entropy += p * log(p)
    # Bigger is worse
    return neg_entropy


# Bigger is worse
quality = quality_by_entropy
#quality = quality_by_max_bucket


def greedy_guess(possibilities):
    best_guess = None
    best_max_confusion = None  # "infinity"
    for i, guess in enumerate(gen_combinations()):
        results_to_possibility_count = Counter()
        for possibility in possibilities:
            result = evaluate(possibility, guess)
            results_to_possibility_count[result] += 1
        max_confusion = quality(results_to_possibility_count)
        maybe_correct = (SLOTS, 0) in results_to_possibility_count
        if best_max_confusion is None or best_max_confusion > max_confusion or (best_max_confusion >= max_confusion and maybe_correct):
            best_max_confusion = max_confusion
            best_guess = guess
        if i > 0 and i % 1000 == 0:
            print(f"    iter {i} of MAX: {best_guess=} {best_max_confusion=}")
    return best_guess, best_max_confusion


def depth_of(possibilities, initial_guess=None):
    if initial_guess is None:
        guess, _ = greedy_guess(possibilities)
    else:
        guess = initial_guess
    by_result = defaultdict(list)
    for possibility in possibilities:
        result = evaluate(possibility, guess)
        by_result[result].append(possibility)
    if (SLOTS, 0) in by_result:
        del by_result[(SLOTS, 0)]
        if not by_result:
            return 1
    else:
        assert len(by_result) > 0 or sum(len(l) for l in by_result.values()) < len(possibilities)
    return 1 + max(depth_of(l) for l in by_result.values())


def run():
    possibilities = list(gen_combinations())
    print(len(possibilities))
    possibilities = apply_restriction(possibilities, "RYGBO", 0, 2)
    possibilities = apply_restriction(possibilities, "SOCCY", 0, 2)
    possibilities = apply_restriction(possibilities, "OSLLB", 1, 3)
    possibilities = apply_restriction(possibilities, "GSLOS", 1, 2)
    possibilities = apply_restriction(possibilities, "OBSOL", 3, 1)
    #possibilities = apply_restriction(possibilities, "RYGB", 0, 3)
    #possibilities = apply_restriction(possibilities, "LBRG", 2, 1)
    #possibilities = apply_restriction(possibilities, "YOLG", 1, 0)
    print(len(possibilities))
    print(possibilities[:10])
    best_guess, best_max_confusion = greedy_guess(possibilities)
    print(f"You should guess {''.join(best_guess)}, which has a maximum quality of {best_max_confusion}.")
    #depth = depth_of(possibilities)
    #print(f"The game will take at most {depth} guesses.")


if __name__ == "__main__":
    test_evaluate()
    run()
