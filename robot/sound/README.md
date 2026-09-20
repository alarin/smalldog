# Sound

A speaker on the Orange Pi's 3.5 mm jack through a small amplifier (2026-09-20). One speaker,
on the amp's left channel; both scripts put everything on L.

| file | what |
|---|---|
| `say` | `say "text"` — espeak-ng through the speaker. Installed on the robot as `/usr/local/bin/say` |
| `bark` | `bark [name] [seconds]` — plays `name.ogg` from this directory. Installed as `/usr/local/bin/bark` |
| `bark.ogg` | "Barking of a dog 2" by Amada44, Wikimedia Commons, CC BY-SA 3.0 |

Install on the robot (`~/smalldog` is the clone):

```bash
sudo apt install espeak-ng ffmpeg alsa-utils
sudo install -m755 robot/sound/say /usr/local/bin/
sudo ln -sf ~/smalldog/robot/sound/bark /usr/local/bin/bark
for c in Headphone Speaker "hp switch" "spk switch" "Output 1" "Output 2" "Left Mixer Left" "Right Mixer Right"; do amixer -c 2 sset "$c" on; done
amixer -c 2 sset PCM 100%; amixer -c 2 sset "Output 1" 100%; amixer -c 2 sset "Output 2" 100%; sudo alsactl store
```

Synthesised barks (formant synthesis, three sizes of dog, a howl) were tried first and sounded
like opera; a recording is the answer. Loudness is the amp's limit: the codec is at full scale.
