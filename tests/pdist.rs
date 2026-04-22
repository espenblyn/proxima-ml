use approx::assert_abs_diff_eq;
use proxima_ml::condensed_index;
use proxima_ml::{DistanceExt, Euclidean, Manhattan};

#[test]
fn pdist_basic() {
    let points = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0, 0.0]];
    let result = Euclidean::pdist(&points);
    assert_eq!(result.len(), 3);
    assert_abs_diff_eq!(result[0], 5.0);
    assert_abs_diff_eq!(result[1], 1.0);
    assert_abs_diff_eq!(result[2], (4.0_f64 + 16.0).sqrt());
}

#[test]
fn pdist_correct_length() {
    let points: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0.0]).collect();
    let result = Euclidean::pdist(&points);
    assert_eq!(result.len(), 10 * 9 / 2);
}

#[test]
fn pdist_matches_pairwise() {
    let points = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![0.0, 1.0, 0.0],
    ];
    let full = Euclidean::pairwise_distances(&points);
    let condensed = Euclidean::pdist(&points);
    let n = points.len();

    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            assert_abs_diff_eq!(condensed[idx], full[i][j], epsilon = 1e-10);
            idx += 1;
        }
    }
}

#[test]
fn pdist_single_vector() {
    let points = vec![vec![1.0, 2.0, 3.0]];
    let result = Euclidean::pdist(&points);
    assert!(result.is_empty());
}

#[test]
fn pdist_two_vectors() {
    let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
    let result = Euclidean::pdist(&points);
    assert_eq!(result.len(), 1);
    assert_abs_diff_eq!(result[0], 5.0);
}

#[test]
fn pdist_works_with_manhattan() {
    let points = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0, 1.0]];
    let result = Manhattan::pdist(&points);
    assert_eq!(result.len(), 3);
    assert_abs_diff_eq!(result[0], 7.0);
    assert_abs_diff_eq!(result[1], 2.0);
    assert_abs_diff_eq!(result[2], 5.0);
}

#[test]
fn pdist_works_with_f32() {
    let points: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
    let result = Euclidean::pdist(&points);
    assert_abs_diff_eq!(result[0], 5.0_f32);
}

#[test]
fn condensed_index_basic() {
    assert_eq!(condensed_index(4, 0, 1), 0);
    assert_eq!(condensed_index(4, 0, 2), 1);
    assert_eq!(condensed_index(4, 0, 3), 2);
    assert_eq!(condensed_index(4, 1, 2), 3);
    assert_eq!(condensed_index(4, 1, 3), 4);
    assert_eq!(condensed_index(4, 2, 3), 5);
}

#[test]
fn condensed_index_swaps_order() {
    assert_eq!(condensed_index(4, 3, 1), condensed_index(4, 1, 3));
    assert_eq!(condensed_index(5, 4, 0), condensed_index(5, 0, 4));
}

#[test]
fn condensed_index_roundtrip() {
    let points = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
        vec![9.0, 10.0],
    ];
    let condensed = Euclidean::pdist(&points);
    let n = points.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let idx = condensed_index(n, i, j);
            let expected = Euclidean::distance(points[i].as_slice(), points[j].as_slice());
            assert_abs_diff_eq!(condensed[idx], expected, epsilon = 1e-10);
        }
    }
}
