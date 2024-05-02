# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import json
import torch
from model import PromptGenerator
from cog import BasePredictor, Input, BaseModel

class PredictorOutput(BaseModel):
    output_path: str = ""
    magic_prompt: str = ""
    logits: list = []
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        assert os.path.exists("config.json"), "config.json not found"
        with open("config.json", "r") as f:
            config = json.load(f)
        self.model = PromptGenerator(config)

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate a magic prompt"),
        output_path: str = Input(
            description="Path to save the generated prompt",
            default="output.txt",
        ),
    ) -> PredictorOutput:
        """Run a single prediction on the model"""
        o_prompt, o_logits = self.model(prompt, output_path)
        return PredictorOutput(output_path=output_path, magic_prompt=o_prompt)
