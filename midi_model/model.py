from transformers import OPTForCausalLM, OPTConfig, AutoModel
import torch

from miditok import TokSequence


class OPTForMusicGeneration(OPTForCausalLM):
    
    def generate_music(self, tokenizer, **kwargs):
        input = torch.tensor([[self.config.bos_token_id]], device=self.device)
        midi = self.generate(input, **kwargs)
        generated_ts = TokSequence(ids=midi.tolist()[0], ids_bpe_encoded=True)
        generated_score = tokenizer(generated_ts)
        return generated_score


OPTForMusicGeneration.register_for_auto_class("AutoModel")
