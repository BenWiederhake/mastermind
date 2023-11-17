extern crate phf_mut;

use phf_mut::{Map, PerfectHash};
use std::cmp::{Ord, Ordering, min};

const MAX_COLS: usize = 8;

fn gauss(n: usize) -> usize {
    (n + 1) * n / 2
}

fn upper_triangle_size(rows_before: usize, cols: usize) -> usize {
    debug_assert!(cols >= rows_before);
    rows_before * (cols - rows_before) + gauss(rows_before)
}

struct MaxSumPairs {
    max_sum: usize,
}

impl MaxSumPairs {
    pub fn new(max_sum: usize) -> Self {
        MaxSumPairs { max_sum }
    }
}

impl PerfectHash for MaxSumPairs {
    type K = (usize, usize);

    fn hash(&self, k: Self::K) -> usize {
        k.0 + upper_triangle_size(k.1, self.max_sum + 1)
    }

    fn size(&self) -> usize {
        1 + self.hash((0, self.max_sum))
    }
}

fn combination_count(slots: usize, cols: usize) -> usize {
    cols.pow(slots as u32)
}

fn try_increment(combo: &mut [u8], cols: usize) -> bool {
    for element in combo.iter_mut() {
        *element += 1;
        if *element as usize != cols {
            return true;
        }
        *element = 0;
    }
    false
}

#[derive(Clone, Debug)]
struct ComboVec {
    slots: usize,
    data: Vec<u8>,
}
impl ComboVec {
    fn new(slots: usize) -> ComboVec {
        ComboVec {
            slots,
            data: vec![],
        }
    }

    fn all(slots: usize, cols: usize) -> ComboVec {
        let mut combovec = ComboVec::new(slots);
        combovec.data.reserve_exact(combination_count(slots, cols));

        let mut combo = vec![0; slots];
        loop {
            combovec.data.extend(&combo);
            if !try_increment(&mut combo, cols) {
                break;
            }
        }

        let expected = combination_count(slots, cols);
        debug_assert_eq!(combovec.data.len(), slots * expected);
        assert_eq!(combovec.len(), expected);
        combovec
    }

    fn len(&self) -> usize {
        let len = self.data.len();
        assert_eq!(len % self.slots, 0);
        len / self.slots
    }

    fn entry(&self, index: usize) -> &[u8] {
        &self.data[(index * self.slots)..((index + 1) * self.slots)]
    }

    fn retain_matches(&mut self, code: &[u8], expected_result: (usize, usize)) {
        for index in (0..self.len()).rev() {
            let actual_result = evaluate_match(self.entry(index), code);
            if actual_result == expected_result {
                continue;
            }
            // Swap-remove from the end. Note that this must operate in reverse, because we cannot
            // control the order in which that the last entry is read.
            for deletion_offset in ((index * self.slots)..((index + 1) * self.slots)).rev() {
                self.data.swap_remove(deletion_offset);
            }
        }
    }

    fn append(&mut self, entry: &[u8]) {
        assert_eq!(entry.len(), self.slots, "{} != {}", entry.len(), self.slots);
        self.data.extend(entry);
    }

    fn classify_buckets(&self, code: &[u8]) -> Map<ComboVec, MaxSumPairs> {
        let mut buckets =
            Map::from_element(MaxSumPairs::new(self.slots), &ComboVec::new(self.slots));
        for index in 0..self.len() {
            let entry = self.entry(index);
            let classification = evaluate_match(entry, code);
            buckets[classification].append(entry);
        }
        buckets
    }

    fn classify_counts(&self, code: &[u8]) -> Map<usize, MaxSumPairs> {
        let mut buckets = Map::new(MaxSumPairs::new(self.slots));
        for index in 0..self.len() {
            let entry = self.entry(index);
            let classification = evaluate_match(entry, code);
            buckets[classification] += 1;
        }
        buckets
    }

    fn iter(&self) -> ComboVecIter {
        ComboVecIter {
            combovec: &self,
            index: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct ComboVecIter<'a> {
    combovec: &'a ComboVec,
    index: usize,
}
impl<'a> Iterator for ComboVecIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.combovec.len() {
            None
        } else {
            let use_index = self.index;
            self.index += 1;
            Some(self.combovec.entry(use_index))
        }
    }
}

fn compute_map_entropy(map: &Map<usize, MaxSumPairs>) -> f32 {
    let n = map.values().sum::<usize>() as f32;
    map.values()
        .map(|count| {
            if *count == 0 {
                0.0
            } else {
                let p = (*count as f32) / n;
                -p * p.log2()
            }
        })
        .sum()
}

#[derive(PartialEq, Debug)]
struct Guess {
    guess_index: usize,
    possible_answer: bool,
    entropy: f32,
}
impl Eq for Guess {
}
impl PartialOrd for Guess {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let order = self.entropy.partial_cmp(&other.entropy).expect("nan entropy");
        if order != Ordering::Equal {
            return Some(order);
        }
        if self.possible_answer != other.possible_answer {
            return match self.possible_answer {
                true => Some(Ordering::Greater),
                false => Some(Ordering::Less),
            };
        }
        // Inverse the order of the index: Want to choose the first index.
        Some(other.guess_index.cmp(&self.guess_index))
    }
}
impl Ord for Guess {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("is always some")
    }
}

fn compute_best_guess(possible_guesses: &ComboVec, possible_solutions: &ComboVec) -> Guess {
    assert!(possible_guesses.len() != 0);
    possible_guesses.iter().enumerate().map(|(index, guess)| {
        let buckets = possible_solutions.classify_counts(guess);
        let perfect_answers = buckets[(possible_guesses.slots, 0)];
        assert!(perfect_answers <= 1, "{}", perfect_answers);
        let entropy = compute_map_entropy(&buckets);
        Guess {
            guess_index: index,
            possible_answer: perfect_answers > 0,
            entropy,
        }
    }).max().unwrap()
}

fn evaluate_match(code1: &[u8], code2: &[u8]) -> (usize, usize) {
    assert_eq!(code1.len(), code2.len());
    let mut correct = 0;
    let mut stray_counts_code1 = [0; MAX_COLS];
    let mut stray_counts_code2 = [0; MAX_COLS];
    for (index, c1) in code1.iter().enumerate() {
        let c2 = code2[index];
        if *c1 == c2 {
            correct += 1;
        } else {
            stray_counts_code1[*c1 as usize] += 1;
            stray_counts_code2[c2 as usize] += 1;
        }
    }
    let mut badloc = 0;
    for (index, s1) in stray_counts_code1.iter().enumerate() {
        let s2 = stray_counts_code2[index];
        badloc += min(s1, &s2);
    }
    assert!(correct + badloc <= code1.len());
    (correct, badloc)
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combination_count() {
        assert_eq!(32768, combination_count(5, 8));
        assert_eq!(1296, combination_count(4, 6));
        assert_eq!(9, combination_count(2, 3));
    }

    #[test]
    fn test_combination_gen_2_3() {
        let combovec = ComboVec::all(2, 3);
        assert_eq!(
            &combovec.data,
            &[0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2],
        );
    }

    #[test]
    fn test_combination_gen_3_2() {
        let combovec = ComboVec::all(3, 2);
        assert_eq!(
            &combovec.data,
            &[0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        );
    }

    #[test]
    fn test_combination_gen_5_8() {
        let combovec = ComboVec::all(5, 8);
        assert_eq!(combovec.entry(0), &[0, 0, 0, 0, 0]);
        assert_eq!(combovec.entry(1), &[1, 0, 0, 0, 0]);
        assert_eq!(combovec.entry(7), &[7, 0, 0, 0, 0]);
        assert_eq!(combovec.entry(8), &[0, 1, 0, 0, 0]);
        assert_eq!(combovec.entry(64), &[0, 0, 1, 0, 0]);
        assert_eq!(combovec.entry(65), &[1, 0, 1, 0, 0]);
        assert_eq!(combovec.entry(512), &[0, 0, 0, 1, 0]);
        assert_eq!(combovec.entry(512 + 24 + 6), &[6, 3, 0, 1, 0]);
    }

    fn test_evaluate_single(c1: &[u8], c2: &[u8], expect: (usize, usize)) {
        let actual1 = evaluate_match(c1, c2);
        let actual2 = evaluate_match(c2, c1);
        assert_eq!(actual1, actual2, "{:?}", (c1, c2, actual1, actual2, expect));
        assert_eq!(actual1, expect, "{:?}", (c1, c2, actual1, actual2, expect));
    }

    #[test]
    fn test_evaluate() {
        test_evaluate_single(&[0, 1, 2, 3], &[0, 1, 2, 3], (4, 0));
        test_evaluate_single(&[0, 1, 2, 3], &[0, 1, 2, 4], (3, 0));
        test_evaluate_single(&[0, 1, 2, 3], &[1, 2, 3, 0], (0, 4));
        test_evaluate_single(&[0, 1, 2, 3], &[1, 2, 4, 4], (0, 2));
        test_evaluate_single(&[0, 1, 2, 3], &[1, 2, 4, 5], (0, 2));
        test_evaluate_single(&[0, 1, 2, 3], &[4, 4, 4, 4], (0, 0));
        test_evaluate_single(&[4, 4, 0, 1], &[4, 4, 1, 2], (2, 1));
        test_evaluate_single(&[4, 4, 0, 1], &[4, 4, 2, 3], (2, 0));
    }

    #[test]
    fn test_combovec_retain_3_2_simple() {
        let mut combovec = ComboVec::all(3, 2);
        combovec.retain_matches(&[0, 0, 1], (3, 0));
        assert_eq!(&combovec.data, &[0, 0, 1]);
    }

    #[test]
    fn test_combovec_retain_3_2_simple2() {
        let mut combovec = ComboVec::all(3, 2);
        combovec.retain_matches(&[0, 0, 1], (2, 0));
        assert_eq!(&combovec.data, &[0, 0, 0, 1, 0, 1, 0, 1, 1]);
    }

    #[test]
    fn test_combovec_retain_5_8_simple2() {
        let mut combovec = ComboVec::all(5, 8);
        combovec.retain_matches(&[0, 1, 2, 3, 4], (4, 1));
        assert_eq!(&combovec.data, &[]);
    }

    #[test]
    fn test_upper_triangle() {
        assert_eq!(upper_triangle_size(0, 0), 0);

        assert_eq!(upper_triangle_size(0, 1), 0);
        assert_eq!(upper_triangle_size(1, 1), 1);

        assert_eq!(upper_triangle_size(0, 2), 0);
        assert_eq!(upper_triangle_size(1, 2), 2);
        assert_eq!(upper_triangle_size(2, 2), 3);

        assert_eq!(upper_triangle_size(0, 3), 0);
        assert_eq!(upper_triangle_size(1, 3), 3);
        assert_eq!(upper_triangle_size(2, 3), 5);
        assert_eq!(upper_triangle_size(3, 3), 6);

        assert_eq!(upper_triangle_size(0, 4), 0);
        assert_eq!(upper_triangle_size(1, 4), 4);
        assert_eq!(upper_triangle_size(2, 4), 7);
        assert_eq!(upper_triangle_size(3, 4), 9);
        assert_eq!(upper_triangle_size(4, 4), 10);

        assert_eq!(upper_triangle_size(0, 5), 0);
        assert_eq!(upper_triangle_size(1, 5), 5);
        assert_eq!(upper_triangle_size(2, 5), 9);
        assert_eq!(upper_triangle_size(3, 5), 12);
        assert_eq!(upper_triangle_size(4, 5), 14);
        assert_eq!(upper_triangle_size(5, 5), 15);
    }

    #[test]
    fn test_phf_max_sum_pairs_0() {
        let hash = MaxSumPairs::new(0);
        assert_eq!(hash.size(), 1);
        assert_eq!(hash.hash((0, 0)), 0);
    }

    #[test]
    fn test_phf_max_sum_pairs_1() {
        let hash = MaxSumPairs::new(1);
        assert_eq!(hash.size(), 3);
        assert_eq!(hash.hash((0, 0)), 0);
        assert_eq!(hash.hash((1, 0)), 1);
        assert_eq!(hash.hash((0, 1)), 2);
    }

    #[test]
    fn test_phf_max_sum_pairs_2() {
        let hash = MaxSumPairs::new(2);
        assert_eq!(hash.size(), 6);
        assert_eq!(hash.hash((0, 0)), 0);
        assert_eq!(hash.hash((1, 0)), 1);
        assert_eq!(hash.hash((2, 0)), 2);
        assert_eq!(hash.hash((0, 1)), 3);
        assert_eq!(hash.hash((1, 1)), 4);
        assert_eq!(hash.hash((0, 2)), 5);
    }

    #[test]
    fn test_phf_max_sum_pairs_3() {
        let hash = MaxSumPairs::new(3);
        assert_eq!(hash.size(), 10);
        assert_eq!(hash.hash((0, 0)), 0);
        assert_eq!(hash.hash((1, 0)), 1);
        assert_eq!(hash.hash((2, 0)), 2);
        assert_eq!(hash.hash((3, 0)), 3);
        assert_eq!(hash.hash((0, 1)), 4);
        assert_eq!(hash.hash((1, 1)), 5);
        assert_eq!(hash.hash((2, 1)), 6);
        assert_eq!(hash.hash((0, 2)), 7);
        assert_eq!(hash.hash((1, 2)), 8);
        assert_eq!(hash.hash((0, 3)), 9);
    }

    #[test]
    fn test_classify_buckets_2() {
        let all = ComboVec::all(2, 2);

        let buckets = all.classify_buckets(&[0, 1]);
        assert_eq!(buckets.len(), 6);
        assert_eq!(buckets[(0, 0)].data, &[]);
        assert_eq!(buckets[(1, 0)].data, &[0, 0, 1, 1]);
        assert_eq!(buckets[(2, 0)].data, &[0, 1]);
        assert_eq!(buckets[(0, 1)].data, &[]);
        assert_eq!(buckets[(1, 1)].data, &[]);
        assert_eq!(buckets[(0, 2)].data, &[1, 0]);

        let buckets = all.classify_buckets(&[0, 0]);
        assert_eq!(buckets.len(), 6);
        assert_eq!(buckets[(0, 0)].data, &[1, 1]);
        assert_eq!(buckets[(1, 0)].data, &[1, 0, 0, 1]);
        assert_eq!(buckets[(2, 0)].data, &[0, 0]);
        assert_eq!(buckets[(0, 1)].data, &[]);
        assert_eq!(buckets[(1, 1)].data, &[]);
        assert_eq!(buckets[(0, 2)].data, &[]);
    }

    #[test]
    fn test_classify_counts_2() {
        let all = ComboVec::all(2, 2);

        let buckets = all.classify_counts(&[0, 1]);
        assert_eq!(buckets.len(), 6);
        assert_eq!(buckets[(0, 0)], 0);
        assert_eq!(buckets[(1, 0)], 2);
        assert_eq!(buckets[(2, 0)], 1);
        assert_eq!(buckets[(0, 1)], 0);
        assert_eq!(buckets[(1, 1)], 0);
        assert_eq!(buckets[(0, 2)], 1);

        let buckets = all.classify_counts(&[0, 0]);
        assert_eq!(buckets.len(), 6);
        assert_eq!(buckets[(0, 0)], 1);
        assert_eq!(buckets[(1, 0)], 2);
        assert_eq!(buckets[(2, 0)], 1);
        assert_eq!(buckets[(0, 1)], 0);
        assert_eq!(buckets[(1, 1)], 0);
        assert_eq!(buckets[(0, 2)], 0);
    }

    #[test]
    fn test_classify_counts_4() {
        let all = ComboVec::all(4, 6);

        let buckets = all.classify_counts(&[0, 1, 2, 3]);
        println!("{:?}", &buckets);
        assert_eq!(buckets.len(), 15);
        assert_eq!(buckets[(0, 0)], 16);
        assert_eq!(buckets[(1, 0)], 108);
        assert_eq!(buckets[(2, 0)], 96);
        assert_eq!(buckets[(3, 0)], 20);
        assert_eq!(buckets[(4, 0)], 1);
        assert_eq!(buckets[(0, 1)], 152);
        assert_eq!(buckets[(1, 1)], 252);
        assert_eq!(buckets[(2, 1)], 48);
        assert_eq!(buckets[(3, 1)], 0);
        assert_eq!(buckets[(0, 2)], 312);
        assert_eq!(buckets[(1, 2)], 132);
        assert_eq!(buckets[(2, 2)], 6);
        assert_eq!(buckets[(0, 3)], 136);
        assert_eq!(buckets[(1, 3)], 8);
        assert_eq!(buckets[(0, 4)], 9);

        let buckets = all.classify_counts(&[0, 0, 1, 1]);
        println!("{:?}", &buckets);
        assert_eq!(buckets.len(), 15);
        assert_eq!(buckets[(0, 0)], 256);
        assert_eq!(buckets[(1, 0)], 256);
        assert_eq!(buckets[(2, 0)], 114);
        assert_eq!(buckets[(3, 0)], 20);
        assert_eq!(buckets[(4, 0)], 1);
        assert_eq!(buckets[(0, 1)], 256);
        assert_eq!(buckets[(1, 1)], 208);
        assert_eq!(buckets[(2, 1)], 32);
        assert_eq!(buckets[(3, 1)], 0);
        assert_eq!(buckets[(0, 2)], 96);
        assert_eq!(buckets[(1, 2)], 36);
        assert_eq!(buckets[(2, 2)], 4);
        assert_eq!(buckets[(0, 3)], 16);
        assert_eq!(buckets[(1, 3)], 0);
        assert_eq!(buckets[(0, 4)], 1);
    }

    // TODO: Test compute_map_entropy

    #[test]
    fn test_best_move_2_3() {
        let all = ComboVec::all(2, 3);
        let best_guess = compute_best_guess(&all, &all);
        let best_combo = all.entry(best_guess.guess_index);
        println!("{:?} -> {:?}", best_guess, best_combo);
        assert_eq!(best_guess, Guess { guess_index: 1, possible_answer: true, entropy: 2.0588138 });
        assert_eq!(best_combo, &[1, 0]);
    }

    #[test]
    fn test_best_move_4_6() {
        let all = ComboVec::all(4, 6);
        let best_guess = compute_best_guess(&all, &all);
        let best_combo = all.entry(best_guess.guess_index);
        println!("{:?} -> {:?}", best_guess, best_combo);
        assert_eq!(best_guess, Guess { guess_index: 51, possible_answer: true, entropy: 3.056671 });
        assert_eq!(best_combo, &[3, 2, 1, 0]);
    }

    #[test]
    fn test_best_move_5_8() {
        let all = ComboVec::all(5, 8);
        let best_guess = compute_best_guess(&all, &all);
        let best_combo = all.entry(best_guess.guess_index);
        println!("{:?} -> {:?}", best_guess, best_combo);
        assert_eq!(best_guess, Guess { guess_index: 83, possible_answer: true, entropy: 3.238308 });
        assert_eq!(best_combo, &[3, 2, 1, 0, 0]);
    }
}
