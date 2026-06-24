from post_tonal.theory.pcset import (
    interval_vector,
    interval_vector_distance,
    invert,
    normalize_pcset,
    pcset_coverage,
    prime_form,
    transpose,
)


def test_normalize_transpose_invert():
    assert normalize_pcset([12, 13, 1, -1]) == [0, 1, 11]
    assert transpose([0, 1, 4], 5) == [5, 6, 9]
    assert invert([0, 1, 4], 0) == [0, 8, 11]


def test_interval_vectors_known_small_sets():
    assert interval_vector([0, 1, 4]) == [1, 0, 1, 1, 0, 0]
    assert interval_vector([0, 1, 4, 6]) == [1, 1, 1, 1, 1, 1]
    assert interval_vector([0, 2, 3, 5]) == [1, 2, 2, 0, 1, 0]


def test_prime_form_and_coverage():
    assert prime_form([0, 1, 4]) == [0, 1, 4]
    assert 0.0 <= interval_vector_distance([0, 1, 4], [0, 2, 3]) <= 6.0
    assert pcset_coverage([0, 1, 7], [0, 1, 4]) == 2 / 3
