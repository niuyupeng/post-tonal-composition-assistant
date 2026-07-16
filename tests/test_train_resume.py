from post_tonal.train import _merge_resume_provenance


def test_resume_provenance_keeps_authoritative_historical_peaks():
    checkpoint = {
        "resume_count": 1,
        "elapsed_seconds": 12.0,
        "peak_process_ram_gib": 0.5,
        "peak_cuda_memory_allocated_gib": 1.25,
        "started_at": "checkpoint-start",
    }
    summary = {
        "resume_count": 2,
        "elapsed_seconds": 14.0,
        "peak_process_ram_gib": 0.75,
        "peak_cuda_memory_allocated_gib": 1.0,
        "started_at": "summary-start",
    }

    merged = _merge_resume_provenance(checkpoint, summary)

    assert merged == {
        "resume_count": 3,
        "elapsed_seconds": 14.0,
        "peak_process_ram_gib": 0.75,
        "peak_cuda_memory_allocated_gib": 1.25,
        "started_at": "summary-start",
    }
