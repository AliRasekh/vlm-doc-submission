# Training-subset example (not an evaluation example)

**Role:** documented real supervised training input/target from `docvqa_train_subset_v1`.  
**Not** part of smoke / development / holdout galleries.

| Field | Value |
|-------|-------|
| question_id | `10196` |
| doc_id | `3338` |
| ucsf_document_id | `jkbg0024` |
| ucsf_document_page_no | `3` |
| image_relpath | `data/images/docvqa_train_subset_v1/qid_10196.png` |
| source_image_sha256 | `6bfec78e2ed30489…` |
| group_key_type | `UCSF` |
| source_shard | `data/train-00006-of-00038.parquet` |

**Question**

What is the Title of the document ?

**Supervised target** (`first_nonempty_source_listed_answer`)

`Agenda`

**All source-listed answers** (preserved in local sidecar; not used for selection beyond the target rule)

["Agenda", "AGENDA"]

**Provenance**

- Dataset: `HuggingFaceM4/DocumentVQA` revision `a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4` (train split)
- Manifest: `data/manifests/docvqa_train_subset_v1.json` sha256 `79fe9acaa0fc6ed3…`
- Seed: `42`; target_n: `2000`; selected: `2000`
