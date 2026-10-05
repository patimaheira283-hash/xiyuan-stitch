from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import load_config
from .run_io import write_json


def main():
    parser = argparse.ArgumentParser(description="下载并锁定扩散修复模型版本")
    parser.add_argument("--config")
    parser.add_argument("--output-config", default="configs/model.lock.yaml")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    config = load_config(args.config)
    ai = config["inpainting"]
    info = HfApi().model_info(ai["model_id"], revision=ai.get("revision", "main"), files_metadata=True)
    ai["revision"] = info.sha
    wanted = [
        "model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
        "safety_checker/config.json", "safety_checker/model.fp16.safetensors",
        "README.md", "LICENSE*",
    ]
    required = [p for p in wanted if p.endswith(".safetensors")]
    missing = set(required) - {item.rfilename for item in info.siblings}
    if missing:
        raise ValueError(f"模型缺少 fp16 safetensors 权重：{sorted(missing)}")
    ai["variant"] = "fp16"
    if not args.metadata_only:
        snapshot_download(ai["model_id"], revision=info.sha,
                          cache_dir=ai["cache_dir"], allow_patterns=wanted)
    additional = []
    from .inpainting import CONTROLNET_MODELS
    mode=ai.get("controlnet","none")
    models=[]
    if mode in CONTROLNET_MODELS:
        model_id,revision,safe_weights=CONTROLNET_MODELS[mode]
        models.append(("controlnet",model_id,revision,["config.json","diffusion_pytorch_model.fp16.safetensors" if safe_weights else "diffusion_pytorch_model.bin","README.md","LICENSE*"]))
    if ai.get("use_lcm"):
        models.append(("lcm",ai["lcm_model_id"],ai["lcm_revision"],["pytorch_lora_weights.safetensors","README.md","LICENSE*"]))
    for kind,model_id,revision,patterns in models:
        model_info=HfApi().model_info(model_id,revision=revision)
        if not args.metadata_only:
            snapshot_download(model_id,revision=model_info.sha,cache_dir=ai["cache_dir"],allow_patterns=patterns)
        additional.append({"kind":kind,"model_id":model_id,"revision":model_info.sha,
                           "license":(model_info.card_data or {}).get("license"),"weights_downloaded":not args.metadata_only})
    ai["local_files_only"] = True
    path = Path(args.output_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(path.with_suffix(".provenance.json"), {
        "model_id": ai["model_id"], "revision": info.sha,
        "url": f"https://huggingface.co/{ai['model_id']}/tree/{info.sha}",
        "license": (info.card_data or {}).get("license"),
        "weights_downloaded": not args.metadata_only,
        "files": [{"path": item.rfilename, "size": item.size} for item in info.siblings],
        "additional_models": additional,
    })
    print(f"模型版本已锁定：{info.sha}\n配置：{path.resolve()}")


if __name__ == "__main__":
    main()
