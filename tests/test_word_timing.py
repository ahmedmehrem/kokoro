from pprint import pprint
import soundfile as sf
from typing import List, cast
import torch
from kokoro.pipeline import Duration, KPipeline, WordTiming

SAMPLE_RATE = 24000


def test_alignment_matrix():
    test_text = "Palestine, officially the State of Palestine, is a country in West Asia. Recognized by 147 of the UN's 193 member states, it encompasses the Israeli-occupied West Bank, including East Jerusalem, and the Gaza Strip, collectively known as the occupied Palestinian territories, within the broader geographic and historical Palestine region. Palestine shares most of its borders with Israel, and it borders Jordan to the east and Egypt to the southwest. It has a total land area of 6,020 square kilometres (2,320 sq mi) while its population exceeds five million people. Its proclaimed capital is Jerusalem, while Ramallah serves as its administrative center. Gaza City was its largest city prior to evacuations in 2023."

    pipeline = KPipeline(lang_code="a")
    generator = pipeline(test_text, voice="af_heart")

    audio_idx = 0
    for text, phonemes, audio, words_timing, _ in generator:
        text = cast(str, text)
        phonemes = cast(str, phonemes)
        audio = cast(torch.FloatTensor, audio)
        words_timing = cast(List, words_timing)

        total_words_timing = Duration.merge_all(
            [timing.duration for timing in words_timing]
        ).second()

        audio_duration = len(audio) / SAMPLE_RATE
        sf.write(f"{audio_idx}.wav", audio, SAMPLE_RATE)
        audio_idx += 1

        pprint(words_timing)

        assert total_words_timing > 0
        assert total_words_timing < audio_duration
