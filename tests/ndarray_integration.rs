#[cfg(feature = "ndarray")]
mod tests {
    use ndarray::{Array2, array};
    use proxima_ml::{Cosine, DistanceExt, Euclidean, SimilarityExt};

    #[test]
    fn test_standard_slices() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let dist = Euclidean::distance(a.as_slice(), b.as_slice());
        assert!(dist > 1.41 && dist < 1.42);
    }

    #[test]
    fn test_ndarray_contiguous() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];

        let dist = Euclidean::distance(&a, &b);
        assert!(dist > 1.41 && dist < 1.42);
    }

    #[test]
    fn test_ndarray_batch_distance() {
        let query = array![1.0, 1.0];

        let targets = array![[1.0, 1.0], [0.0, 1.0], [1.0, 0.0],];

        let distances = Euclidean::batch_distance(&query, targets.outer_iter());

        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0], 0.0);
        assert_eq!(distances[1], 1.0);
        assert_eq!(distances[2], 1.0);
    }

    #[test]
    fn test_similarity_mixed_types() {
        let a_slice = vec![1.0, 2.0, 3.0];
        let b_array = array![1.0, 2.0, 3.0];

        let sim = Cosine::similarity(a_slice.as_slice(), &b_array);

        assert!(sim > 0.999);
    }
}
