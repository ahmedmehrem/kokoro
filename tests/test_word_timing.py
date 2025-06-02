from typing import cast
import torch
from kokoro.pipeline import KPipeline

SAMPLE_RATE = 22050
HOP_LENGTH = 600
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE


def test_alignment_matrix():
    test_text = "Hello, World"

    pipeline = KPipeline(lang_code="a")
    generator = pipeline(test_text, voice="af_heart")

    for text, phonemes, audio, align_matrix in generator:
        text = cast(str, text)
        phonemes = cast(str, phonemes)
        audio = cast(torch.FloatTensor, audio)
        align_matrix = cast(torch.FloatTensor, align_matrix)
        num_phonemes = len(phonemes)

        num_frames = align_matrix.shape[2]

        assert num_phonemes == align_matrix.shape[1] - 2
        assert num_frames * FRAME_DURATION == len(audio) / SAMPLE_RATE
