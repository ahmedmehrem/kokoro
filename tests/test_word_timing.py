from typing import List, cast
import torch
from kokoro.pipeline import Duration, KPipeline, WordTiming

SAMPLE_RATE = 22050


def test_alignment_matrix():
    test_text = "Hello, World"

    pipeline = KPipeline(lang_code="a")
    generator = pipeline(test_text, voice="af_heart")

    for text, phonemes, audio, words_timing in generator:
        text = cast(str, text)
        phonemes = cast(str, phonemes)
        audio = cast(torch.FloatTensor, audio)
        words_timing = cast(List[WordTiming], words_timing)

        print(words_timing)

        total_words_timing = Duration.merge_all(
            [timing.duration for timing in words_timing]
        ).second()

        audio_duration = len(audio) / SAMPLE_RATE

        assert total_words_timing > 0
        assert total_words_timing < audio_duration
