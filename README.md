# Kokoro TTS

This fork added the feature to calculate the word timing data out of the model predicted alignment matrix

```python
from kokoro.pipeline import Duration, KPipeline, WordTiming
import soundfile as sf

test_text = "Hello, World"

pipeline = KPipeline(lang_code="a")
generator = pipeline(test_text, voice="af_heart")

for text, phonemes, audio, words_timing in generator:
    sf.write("audio.wav", audio.numpy(), SAMPLE_RATE)
    print(words_timing)

```

```
[WordTiming(word='Hello,', duration=Duration(start=0.46258503401360546, end=0.870748299319728)), WordTiming(word='World', duration=Duration(start=0.9251700680272109, end=1.5510204081632653))]
```
