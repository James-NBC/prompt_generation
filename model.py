import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PromptGenerator:
    def __init__(self, config, device=None):
        self.device = device
        if self.device is None:
            self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["magic_prompt_ckpt"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["magic_prompt_ckpt"]).to(self.device)
        
    def __call__(self, prompt, output_path = None):
        output_logits = None
        if len(prompt) == 0:
            output_prompt = ""
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            if len(input_ids[0]) >= self.config["max_length"]:
                output_prompt = prompt
            else:
                outputs = self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id, max_length = self.config["max_length"], return_dict_in_generate=True, output_logits=True)
                output_prompt = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                output_logits = outputs.logits
        if output_path is not None:
            with open(output_path, 'w') as f:
                f.write(output_prompt)
        torch.cuda.empty_cache()
        return output_prompt, output_logits
