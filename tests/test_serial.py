from post_tonal.theory.serial import (
    I,
    P,
    R,
    RI,
    aggregate_completion_rate,
    cyclic_row_order_accuracy,
    generate_twelve_tone_row,
    is_valid_row,
    row_form,
    row_order_accuracy,
)


def test_row_validity_and_generation():
    row = generate_twelve_tone_row(seed=1)
    assert is_valid_row(row)
    assert not is_valid_row([0, 1, 2])
    assert not is_valid_row(list(range(11)) + [10])


def test_serial_transformations():
    row = list(range(12))
    assert P(row, 3) == [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]
    assert R(row, 0) == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert I(row, 0) == [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert RI(row, 0) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    assert row_form(row, "RI0") == RI(row, 0)


def test_serial_metrics():
    row = list(range(12))
    assert row_order_accuracy(row, row) == 1.0
    assert row_order_accuracy([0, 2, 1], row) == 1 / 12
    assert cyclic_row_order_accuracy(row + row, row) == 1.0
    assert aggregate_completion_rate([0, 1, 2, 3, 4, 5]) == 0.5
