import time
import torch
import argparse
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM



app = Flask(__name__)

def load_llms(ckpt_dir, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="auto", torch_dtype=torch.bfloat16)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    return parser.parse_args()

CKPT_DIR = "checkpoints"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, tokenizer = load_llms(CKPT_DIR, device)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    start = time.time()
    json_request = request.get_json(force=True)
    prompt = json_request['prompt']
    output_path = json_request['output_path']
    if len(prompt) == 0:
        output_prompt = ""
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if len(input_ids[0]) > 25:
            output_prompt = prompt
        else:
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length = 26)
            output_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # write to txt file
    with open(output_path, 'w') as f:
        f.write(output_prompt)
    torch.cuda.empty_cache()
    return jsonify({"output_path": output_path, "time": time.time() - start})  

if __name__ == "__main__":
    args = parse_args()
    app.run(host='127.0.0.1', port = args.port)