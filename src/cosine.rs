use crate::validate_lengths;
use crate::{Distance, Similarity};
use num_traits::Float;
use std::iter::Sum;

pub struct Cosine;

impl<F: Float + Sum> Similarity<F> for Cosine {
    fn compute_similarity(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);

        let (dot, sq_a, sq_b) = a.iter().zip(b.iter()).fold(
            (F::zero(), F::zero(), F::zero()),
            |(dot, sq_a, sq_b), (x, y)| (dot + *x * *y, sq_a + *x * *x, sq_b + *y * *y),
        );

        let mag_a = sq_a.sqrt();
        let mag_b = sq_b.sqrt();

        if mag_a == F::zero() || mag_b == F::zero() {
            return F::zero();
        }

        dot / (mag_a * mag_b)
    }
}

impl<F: Float + Sum> Distance<F> for Cosine {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        F::one() - Self::compute_similarity(a, b)
    }
}
