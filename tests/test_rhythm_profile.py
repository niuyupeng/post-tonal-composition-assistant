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


def test_rhythm_profile_distance_is_voice_count_normalized():
    voice = generate_rhythmic_profile("medium", measures=4, seed=3)
    doubled = [
        {**event, "voice": voice_index}
        for voice_index in range(2)
        for event in voice
    ]
    single_distance = rhythmic_profile_distance(voice, "medium", measures=4, voice_count=1)
    doubled_distance = rhythmic_profile_distance(doubled, "medium", measures=4, voice_count=2)
    assert doubled_distance == single_distance
