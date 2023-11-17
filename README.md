# mastermind

This is a solver for the "mastermind" code-guessing game. There are many like it, but this one is mine.

The solver uses the "entropy" heuristic, which is probably close to optimal,
but definitely not exactly optimal. In particular, it tries to maximize the
entropy it will gather from the answer to its guess. This leads to some
counter-intuitive yet clever behavior, like guessing codes that cannot
possibly be correct, yet yield strictly more information than a
possibly-correct code.

Key results are:
- The "medium" game (with 4 slots and 6 colors) never needs to last more than 6 rounds, and the "optimal" (see above) first guess is "RYGB", i.e. any 4 different colors.
- Wikipedia claims that this can be always done in 5 rounds or fewer :(
- The "large" game (with 5 slots and 8 colors) never needs to last more than 7 rounds, and the "optimal" (see above) first guess is "RRYGB", i.e. any 4 (not 5) different colors.
- Wikipedia does not make a claim here, but I assume this can be improved to 6 rounds.

## Table of Contents

- [Background](#background)
- [Usage](#usage)
- [TODOs](#todos)
- [Contribute](#contribute)

## Background

See the [rules to the board game on Wikipedia](https://en.wikipedia.org/wiki/Mastermind_(board_game)).

## Usage

There is no proper way to use it yet.

For the python program, simply adjust the globals `COLORS` and `slots` to your
liking, and possibly uncomment one of the `possibilities =
apply_restriction(possibilities, "RYGB", 0, 3)` lines, and insert the actual
observations made so far. The program then computes the "best" (see header)
move, and how long the game will take at maximum.

The rust program is not really executable yet. The test cases can be abused, though.

## TODOs

* Some form of interactivity?
* Hardcode the handful of hardest results, with a flag to skip so they can be recomputed?
* Some kind of wasm binary and a website, Like I did with [mebongo](https://benwiederhake.github.io/mebongo/)?
* Maybe some way to find the "optimal" solution?

## Contribute

Feel free to dive in! [Open an issue](https://github.com/BenWiederhake/mastermind/issues/new) or submit PRs.
