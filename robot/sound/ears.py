"""The robot's ears: the camera's USB mic through sherpa-onnx, into two commands.

    for kind, text in Ears():          # ('heard', 'сина гуляй'), ('cmd', 'go'), ...
        ...

Mic: the IMX415 camera module's USB audio (ALSA card "Camera", plughw:3,0); the ES8388's
mic input has nothing wired to it. Recognizer: sherpa-onnx with the streaming Russian
zipformer from alphacephei (vosk-model-small-streaming-ru, int8), 8x real time on four
of the Pi's cores. Model files under ~/models/vosk-small-streaming-ru (README).

Range: about 30 cm, spoken clearly. The mic's noise floor is ~-36 dBFS and a voice at
1 m only clears it by ~6 dB, which no model decodes; at 30 cm it is 15 dB and the small
model gets "сина гуляй" — the "п" of псина is usually lost, so the wake word is matched
fuzzily, the command word is looked for in the same or the next utterance, and "стоп"
alone is enough. `listen` is the command-line front (--dry, the shell actions);
ros2/smalldog_hardware's ears node is the ROS one.
"""
import difflib
import os
import re
import subprocess
import time

MODEL = os.path.expanduser('~/models/vosk-small-streaming-ru')
MIC = os.environ.get('MIC', 'plughw:3,0')
WAKE = ('псина', 'псинка')
CMDS = {'stop': ('стоп', 'стой', 'стоять', 'стопс'),
        'go': ('гуляй', 'гулять', 'гуляет', 'вперёд', 'вперед')}
REPLY = {'stop': 'стою', 'go': 'гуляю'}
WINDOW = 3.0        # s: the wake word and the command may land in two utterances
REPEAT = 2.0        # s: the same command again this soon is an echo of the first


def close(word, targets, cut):
    return any(difflib.SequenceMatcher(None, word, t).ratio() >= cut for t in targets)


def match(words):
    """words of the last WINDOW seconds → 'stop' | 'go' | None. Wake word first, command after
    it; a bare "стоп" counts too — the first word of the first phrase tends to get lost, and a
    stop for nothing is harmless where a walk for nothing is not."""
    for i, w in enumerate(words):
        if close(w, WAKE, 0.6):
            for w2 in words[i + 1:i + 3]:
                for name, ts in CMDS.items():
                    if close(w2, ts, 0.75):
                        return name
    return 'stop' if any(close(w, CMDS['stop'], 0.75) for w in words) else None


def say(text):
    """the robot's reply, blocking: two at once fight for the speaker"""
    try:
        subprocess.run(['say', text], timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


class Ears:
    """iterate for ('partial', text) as words come in, ('heard', text) at each end of an
    utterance and ('cmd', 'stop'|'go') when one matches; `seconds` bounds the run."""

    def __init__(self, mic=MIC, model=MODEL, seconds=None, threads=4):
        import sherpa_onnx
        self.mic, self.seconds = mic, seconds
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=f'{model}/am-onnx/encoder.int8.onnx', decoder=f'{model}/am-onnx/decoder.int8.onnx',
            joiner=f'{model}/am-onnx/joiner.int8.onnx', tokens=f'{model}/lang/tokens.txt',
            num_threads=threads, sample_rate=16000, decoding_method='modified_beam_search',
            max_active_paths=10, enable_endpoint_detection=True, rule1_min_trailing_silence=1.5,
            rule2_min_trailing_silence=0.8, rule3_min_utterance_length=15)
        self.stop = False

    def __iter__(self):
        import numpy as np
        p = subprocess.Popen(['arecord', '-q', '-D', self.mic, '-f', 'S16_LE', '-r', '16000', '-c', '1'],
                             stdout=subprocess.PIPE)
        try:
            p.stdout.read(3200 * 5)                  # the camera's first 0.5 s is a full-scale burst
            s = self.rec.create_stream()
            recent = []                              # (time, word) of the last finals
            fired = (None, 0.0)
            last = ''
            t0 = time.time()
            while not self.stop and (self.seconds is None or time.time() - t0 < self.seconds):
                b = p.stdout.read(3200)
                if not b:
                    break
                s.accept_waveform(16000, np.frombuffer(b, np.int16).astype(np.float32) / 32768)
                while self.rec.is_ready(s):
                    self.rec.decode_stream(s)
                txt = self.rec.get_result(s)
                if txt != last:
                    last = txt
                    yield 'partial', txt
                if not self.rec.is_endpoint(s):
                    continue
                self.rec.reset(s)
                last = ''
                if not txt:
                    continue
                now = time.time()
                yield 'heard', txt
                recent = [(t, w) for t, w in recent if now - t < WINDOW] + \
                         [(now, w) for w in re.findall(r'\w+', txt.lower())]
                cmd = match([w for _, w in recent])
                if cmd is None:
                    continue
                recent = []
                if cmd == fired[0] and now - fired[1] < REPEAT:
                    continue
                fired = (cmd, now)
                yield 'cmd', cmd
        finally:
            p.kill()
