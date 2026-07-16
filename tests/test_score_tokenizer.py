from post_tonal.data.score_tokenizer import ScoreTokenizer
from post_tonal.models.transformer import PostTonalTransformer
from post_tonal.theory.pcset import interval_vector


def _metadata():
    pcset = [0, 1, 4]
    return {
        "pcset": pcset,
        "interval_vector": interval_vector(pcset),
        "row": None,
        "row_form": None,
        "rhythm_profile": "sparse",
        "gesture": "fragmented",
        "voices": 2,
        "measures": 3,
        "instrument": "clarinet",
    }


def test_bar_tokens_preserve_measure_time_and_requested_span():
    tokenizer = ScoreTokenizer()
    metadata = _metadata()
    events = [
        {"onset": 0.5, "duration": 0.25, "voice": 0, "pitch": 60, "pc": 0, "is_rest": False},
        {"onset": 8.25, "duration": 0.5, "voice": 1, "pitch": 64, "pc": 4, "is_rest": False},
    ]
    tokens = tokenizer.events_to_tokens(events, metadata)
    assert tokens.count("BAR") == metadata["measures"]
    decoded = tokenizer.tokens_to_events(tokens, metadata)
    assert [event["onset"] for event in decoded] == [0.5, 8.25]


def test_score_grammar_requires_requested_bar_count_before_eos():
    tokenizer = ScoreTokenizer()
    metadata = _metadata()
    ids = tokenizer.encode(tokenizer.condition_tokens(metadata))
    allowed = tokenizer.allowed_next_token_ids(ids, metadata)
    assert allowed == [tokenizer.token_to_id["BAR"]]

    ids.append(tokenizer.token_to_id["BAR"])
    allowed = tokenizer.allowed_next_token_ids(ids, metadata)
    assert tokenizer.eos_id not in allowed
    assert tokenizer.token_to_id["TIME_SHIFT_0"] in allowed

    ids.extend([tokenizer.token_to_id["BAR"], tokenizer.token_to_id["BAR"]])
    allowed = tokenizer.allowed_next_token_ids(ids, metadata)
    assert tokenizer.eos_id in allowed
    assert tokenizer.token_to_id["BAR"] not in allowed


def test_score_grammar_keeps_onsets_inside_current_measure():
    tokenizer = ScoreTokenizer()
    metadata = _metadata()
    ids = tokenizer.encode(tokenizer.condition_tokens(metadata))
    ids.extend(
        tokenizer.encode(
            [
                "BAR",
                "TIME_SHIFT_15",
                "VOICE_0",
                "PITCH_0",
                "OCTAVE_4",
                "DUR_1",
            ]
        )
    )
    allowed = tokenizer.allowed_next_token_ids(ids, metadata)
    assert tokenizer.token_to_id["TIME_SHIFT_0"] in allowed
    assert tokenizer.token_to_id["TIME_SHIFT_1"] not in allowed
    assert tokenizer.token_to_id["BAR"] in allowed


def test_sampling_window_preserves_condition_prefix():
    model = PostTonalTransformer(
        vocab_size=64,
        hidden_size=24,
        layers=1,
        heads=3,
        max_seq_len=10,
        dropout=0.0,
    )
    sequence = [1, 2, 3] + list(range(10, 30))
    window = model._sampling_window(sequence, prefix_length=3)
    assert len(window) == 10
    assert window[:3] == [1, 2, 3]
    assert window[3:] == sequence[-7:]
