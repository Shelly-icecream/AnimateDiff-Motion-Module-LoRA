# coding=utf-8
# Copyright 2023, Haofan Wang, Qixun Wang, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#  Changes were made to this source code by Yuwei Guo.
""" Conversion script for the LoRA's safetensors checkpoints. """

import argparse

import torch
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline


import torch
import torch.nn as nn

def load_diffusers_lora(pipeline, state_dict, alpha=0.6):
    loaded = 0
    skipped = 0
    visited = set()

    for key in state_dict.keys():

        if "lora_A" not in key:
            continue

        if key in visited:
            continue

        up_key = key.replace("lora_A", "lora_B")

        if up_key not in state_dict:
            print(f"[Warning] Missing pair for {key}")
            continue

        model_key = key.replace(".lora_A", "")
        layer_infos = model_key.split(".")

        curr_layer = pipeline.unet

        # 逐层查找
        for name in layer_infos:
            if hasattr(curr_layer, name):
                curr_layer = getattr(curr_layer, name)
            else:
                raise ValueError(f"Layer {name} not found in UNet")


        if not hasattr(curr_layer, "weight"):
            print(f"[Skip] {model_key} is not nn.Linear but {type(curr_layer)}")
            skipped += 1
            continue
        loaded += 1
        weight_down = state_dict[key].to(torch.float32)
        weight_up   = state_dict[up_key].to(torch.float32)

        if weight_up.shape[1] != weight_down.shape[0]:
            raise ValueError(
                f"Shape mismatch: {weight_up.shape} x {weight_down.shape}"
            )

        delta = torch.mm(weight_up, weight_down)

        if curr_layer.weight.shape != delta.shape:
            raise ValueError(
                f"Delta shape {delta.shape} does not match {curr_layer.weight.shape}"
            )

        curr_layer.weight.data += alpha * delta.to(curr_layer.weight.device)

        visited.add(key)
        visited.add(up_key)
        print("loaded =", loaded, "skipped =", skipped)
        print(f"[Loaded] {model_key}")

    print("✅ Motion LoRA loaded safely.")
    return pipeline



def convert_lora(
    pipeline,
    state_dict,
    alpha=0.6,
):

    visited = set()

    for key in state_dict.keys():

        # 只处理 lora_A
        if "lora_A" not in key:
            continue

        if key in visited:
            continue

        up_key = key.replace("lora_A", "lora_B")

        if up_key not in state_dict:
            print(f"[Warning] Missing pair for {key}")
            continue

        # 去掉 .lora_A
        model_key = key.replace(".lora_A", "")

        layer_infos = model_key.split(".")

        curr_layer = pipeline.unet

        # 逐级定位层
        for name in layer_infos:
            if hasattr(curr_layer, name):
                curr_layer = getattr(curr_layer, name)
            else:
                raise ValueError(f"Layer {name} not found in UNet")

        weight_down = state_dict[key].to(torch.float32)      # [r, in]
        weight_up   = state_dict[up_key].to(torch.float32)   # [out, r]

        # shape sanity check
        if weight_up.shape[1] != weight_down.shape[0]:
            raise ValueError(
                f"Shape mismatch: {weight_up.shape} x {weight_down.shape}"
            )

        delta = torch.mm(weight_up, weight_down)  # [out, in]

        if curr_layer.weight.shape != delta.shape:
            raise ValueError(
                f"Delta shape {delta.shape} does not match layer weight {curr_layer.weight.shape}"
            )

        curr_layer.weight.data += alpha * delta.to(curr_layer.weight.device)

        visited.add(key)
        visited.add(up_key)

        print(f"[Loaded] {model_key} | delta shape: {delta.shape}")

    print("✅ Motion LoRA loaded successfully.")

    return pipeline



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_path", default=None, type=str, required=True, help="Path to the base model in diffusers format."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--lora_prefix_unet", default="lora_unet", type=str, help="The prefix of UNet weight in safetensors"
    )
    parser.add_argument(
        "--lora_prefix_text_encoder",
        default="lora_te",
        type=str,
        help="The prefix of text encoder weight in safetensors",
    )
    parser.add_argument("--alpha", default=0.75, type=float, help="The merging ratio in W = W0 + alpha * deltaW")
    parser.add_argument(
        "--to_safetensors", action="store_true", help="Whether to store pipeline in safetensors format or not."
    )
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")

    args = parser.parse_args()

    base_model_path = args.base_model_path
    checkpoint_path = args.checkpoint_path
    dump_path = args.dump_path
    lora_prefix_unet = args.lora_prefix_unet
    lora_prefix_text_encoder = args.lora_prefix_text_encoder
    alpha = args.alpha

    pipe = convert(base_model_path, checkpoint_path, lora_prefix_unet, lora_prefix_text_encoder, alpha)

    pipe = pipe.to(args.device)
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
