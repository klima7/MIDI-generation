# MIDI-generation

Model based on `facebook/opt-125m` trained on small dataset to generate music from old games in midi format.

## Usage
```
!pip install miditok==3.0.2
from transformers import AutoModel, AutoModelForCausalLM
from miditok import MIDITokenizer
tokenizer = MIDITokenizer.from_pretrained('lklimkiewicz/midi-ganerator-game')
model = AutoModel.from_pretrained('lklimkiewicz/midi-ganerator-game', trust_remote_code=True).cuda()
music = model.generate_music(tokenizer)
music.dump_midi('out.mid')
!timidity out.mid
```
