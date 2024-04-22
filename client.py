import requests
import argparse

API_NAME = "generate_prompt"

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape painting", help="Prompt for the model")
    return parser.parse_args()

def main():
    args = parse_args()
    url = f"http://localhost:{args.port}/{API_NAME}"
    json_request = {
        "prompt": args.prompt,
    }
    response = requests.post(url, json=json_request)
    import json
    with open('result.json', 'w') as f:
        json.dump(response.json(), f)
    print(response.text)

if __name__ == "__main__":
    main()