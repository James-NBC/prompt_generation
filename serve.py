import os
import time
import json
from ray import serve
from model import PromptGenerator
from starlette.requests import Request

@serve.deployment
class RayPromptGenerator:
    def __init__(self, config):
        self.model = PromptGenerator(config)

    async def __call__(self, http_request: Request) -> str:
        start = time.time()
        json_request = await http_request.json()
        prompt = json_request["prompt"]
        output_path = json_request.get("output_path", None)
        generated_prompt = self.model(prompt, output_path)
        return {"output_path": output_path, "prompt": generated_prompt, "inference_time": time.time() - start}

assert os.path.exists("config.json"), "config.json not found"
with open("config.json", "r") as f:
    config = json.load(f)
prompt_generator_app = RayPromptGenerator.bind(config)