use crate::Distance;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Euclidean {}

pub struct SqEuclidean {}

impl<F: Float + Sum> Distance<F> for Euclidean {
    fn distance(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<F>()
            .sqrt()
    }
}

impl<F: Float + Sum> Distance<F> for SqEuclidean {
    fn distance(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<F>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn euclidean_basic() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.0);
    }

    #[test]
    fn euclidean_three_dimensions() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.196152, epsilon = 1e-6);
    }

    #[test]
    fn euclidean_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &a), 0.0);
    }

    #[test]
    fn euclidean_single_element() {
        let a = [5.0];
        let b = [3.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 2.0);
    }

    #[test]
    fn sq_euclidean_basic() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_abs_diff_eq!(SqEuclidean::distance(&a, &b), 25.0);
    }

    #[test]
    fn sq_equals_euclidean_squared() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let euclidean = Euclidean::distance(&a, &b);
        let sq = SqEuclidean::distance(&a, &b);
        assert_abs_diff_eq!(sq, euclidean * euclidean, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "proxima: slice lengths must match")]
    fn euclidean_mismatched_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        Euclidean::distance(&a, &b);
    }

    #[test]
    fn euclidean_works_with_f32() {
        let a: [f32; 2] = [0.0, 0.0];
        let b: [f32; 2] = [3.0, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.0_f32);
    }
}
