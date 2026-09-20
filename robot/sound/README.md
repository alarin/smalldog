# Sound

A speaker on the Orange Pi's 3.5 mm jack through a small amplifier (2026-09-20). One speaker,
on the amp's left channel; both scripts put everything on L.

| file | what |
|---|---|
| `say` | `say "text"` — espeak-ng through the speaker. Installed on the robot as `/usr/local/bin/say` |
| `bark` | `bark [name] [seconds]` — plays `name.ogg` from this directory. Installed as `/usr/local/bin/bark` |
| `bark.ogg` | "Barking of a dog 2" by Amada44, Wikimedia Commons, CC BY-SA 3.0 |
| `ears.py` | the recognizer: the camera mic through sherpa-onnx → `('heard', text)` / `('cmd', 'stop'\|'go')` |
| `listen` | `ears.py` without ROS: «псина, стоп» / «псина, гуляй» → `ros2 topic pub`; `--dry` only prints. Installed as `/usr/local/bin/listen` |

Install on the robot (`~/smalldog` is the clone):

```bash
sudo apt install espeak-ng ffmpeg alsa-utils
sudo install -m755 robot/sound/say /usr/local/bin/
sudo ln -sf ~/smalldog/robot/sound/bark /usr/local/bin/bark
for c in Headphone Speaker "hp switch" "spk switch" "Output 1" "Output 2" "Left Mixer Left" "Right Mixer Right"; do amixer -c 2 sset "$c" on; done
amixer -c 2 sset PCM 100%; amixer -c 2 sset "Output 1" 100%; amixer -c 2 sset "Output 2" 100%; sudo alsactl store
```

## Listening

The mic is the camera module's USB audio (ALSA card `Camera`, `plughw:3,0`); the ES8388's
own mic input has nothing on it. `ears.py` runs sherpa-onnx with alphacephei's streaming
Russian zipformer (the small vosk model, int8: 27 MB, 8× real time on four cores) and
matches the wake word fuzzily — the "п" of «псина» is usually lost; a bare «стоп» counts
without the wake word. Under ROS the ears are `smalldog_hardware ears`
(`robot.launch.py voice:=true`): «стоп» → `/smalldog/explore false` + a zero `/cmd_vel`,
«гуляй» → `/smalldog/explore true`, and the robot answers «стою» / «гуляю». Range is about 30 cm: the mic's noise floor is around −36 dBFS and a
voice at 1 m clears it by only 6 dB, which no model decodes. `listen --dry` prints what
it hears and runs nothing. `robot_explore.sh` launches with `voice:=true` (`VOICE=0` not to).

alphacephei.com times out from here; the model is on Hugging Face:

```bash
pip3 install --user --break-system-packages sherpa-onnx
M=~/models/vosk-small-streaming-ru; mkdir -p $M/am-onnx $M/lang
B=https://huggingface.co/alphacep/vosk-model-small-streaming-ru/resolve/main
for f in am-onnx/encoder.int8.onnx am-onnx/decoder.int8.onnx am-onnx/joiner.int8.onnx lang/tokens.txt; do curl -sSL -o $M/$f $B/$f; done
amixer -c 3 sset Mic 100%; sudo alsactl store
sudo ln -sf ~/smalldog/robot/sound/listen /usr/local/bin/listen
```

Synthesised barks (formant synthesis, three sizes of dog, a howl) were tried first and sounded
like opera; a recording is the answer. Loudness is the amp's limit: the codec is at full scale.
