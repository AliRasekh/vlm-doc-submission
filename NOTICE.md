# Upstream attribution

This submission evaluates publicly released models and datasets. Weights and data
are not redistributed except for the project-trained LoRA adapter under
`artifacts/internvl3_1b_lora_v2_u200/`.

| Asset | Source | Notes |
|-------|--------|-------|
| SmolVLM-256M/500M | Hugging Face `HuggingFaceTB/SmolVLM-*-Instruct` | Pinned revisions in `configs/models/` |
| InternVL3-1B | Hugging Face `OpenGVLab/InternVL3-1B` | Pinned revision `4415a3b8…` |
| DocVQA | `HuggingFaceM4/DocumentVQA` @ `a44195f8…` | Manifests only in this repo |
| Pix2Struct ANLS reference | Google Research Pix2Struct (threshold semantics) | Cited in metric docs; not vendored |
| VLMEvalKit | Open-source eval toolkit | Route/metric reference only |

Respect the upstream licenses of each Hugging Face model card and dataset card
when downloading base weights or DocVQA material.
