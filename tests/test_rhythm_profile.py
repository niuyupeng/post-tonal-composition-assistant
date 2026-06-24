from post_tonal.theory.rhythm_profile import RHYTHM_PROFILES, density_curve, generate_rhythmic_profile, rhythmic_profile_distance


def test_rhythm_profiles_generate_inside_duration():
    for profile in RHYTHM_PROFILES:
        events = generate_rhythmic_profile(profile, measures=4, seed=2)
        assert events
        assert all(0.0 <= float(event["onset"]) < 16.0 for event in events)
        assert all(float(event["duration"]) > 0.0 for event in events)
        assert len(density_curve(events, measures=4)) == 4


def test_rhythm_profile_distance_is_nonnegative():
    events = generate_rhythmic_profile("pointillistic", measures=4, seed=3)
    assert rhythmic_profile_distance(events, "pointillistic", measures=4) >= 0.0
