import argparse
import logging
import time

import torch
import numpy as np
import json
from omegaconf import DictConfig, OmegaConf
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.profiler import profile, record_function, ProfilerActivity

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

class Demo:
    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

def parse_arguments(params=None):
    parser = argparse.ArgumentParser(description="Demo full frame classification...")
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")
    parser.add_argument("--output_json", type=str, default=None, help="The file path for where to output a json of the benchmark results.")
    return parser.parse_args(params)

def main():
    args = parse_arguments()
    conf = OmegaConf.load(args.path_to_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(conf).to(device)
    if conf.model.checkpoint:
        snapshot = torch.load(conf.model.checkpoint, map_location=device)
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval()

    transform = Demo.get_transform_for_inf(conf.test_transforms)
    
    dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    processed_input = transform(image=dummy_input)['image'].unsqueeze(0)
    input_tensor = processed_input.repeat(1, 1, 1, 1).to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True  # Ensure stack tracing is enabled
    ) as prof:
        start_time = time.time()
        with torch.no_grad():
            #with record_function("model_inference"):
                output = model(input_tensor)
        total_time = time.time() - start_time

    avg_inference_time = total_time / 100
    inferences_per_second = 100 / total_time

    print(f"Avg inference time: {avg_inference_time:.3f} seconds per image")
    print(f"Inference per second: {inferences_per_second:.2f} inferences per second")

    # Save the profiling results to a JSON file in the Chrome trace format
    prof.export_chrome_trace("cnn_gesture_profile.json")
    print("Profiler results exported to cnn_gesture_profile.json")

    profiler_results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    print(profiler_results)

    output_data = {
        "avg_inference_time": avg_inference_time,
        "inferences_per_second": inferences_per_second,
        "profiler_results_path": "cnn_gesture_profile.json",
        "profiler_results": profiler_results
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
