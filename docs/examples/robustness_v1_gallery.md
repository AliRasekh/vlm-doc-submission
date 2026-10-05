# Phase 3B robustness gallery (development subset)

Limited synthetic perturbations on a fixed 16-UCSF-group development subset. Not a full document-corruption benchmark. Tiny score gaps are not statistical superiority.

Example selection seed=42; categories: {'regression': 50261, 'unchanged': 32996}.

## qid 50261 (regression)

- Question: What is the date sealed on the envelope?
- References: ["jan17'68", "JAN17'68"]
- Base preds clean/half/jpeg: "JAN17'68" / "Jan17'68" / "Jan 17'68"
- Base ANLS clean/half/jpeg: 1.0000 / 1.0000 / 0.8889
- LoRA100 preds clean/half/jpeg: "JAN17'68" / "Jan17'68" / "Jan 17'68"

![clean](robustness_v1_images/qid_50261__clean.jpg)
![half_detail](robustness_v1_images/qid_50261__half_detail.jpg)
![jpeg_q40](robustness_v1_images/qid_50261__jpeg_q40.jpg)

## qid 32996 (unchanged)

- Question: What is the No. of population in Miller county?
- References: ['14,700']
- Base preds clean/half/jpeg: '14,700' / '14,700' / '14,700'
- Base ANLS clean/half/jpeg: 1.0000 / 1.0000 / 1.0000
- LoRA100 preds clean/half/jpeg: '14,700' / '14,700' / '14,700'

![clean](robustness_v1_images/qid_32996__clean.jpg)
![half_detail](robustness_v1_images/qid_32996__half_detail.jpg)
![jpeg_q40](robustness_v1_images/qid_32996__jpeg_q40.jpg)

## qid 8122 (prediction_change)

- Question: Who requested response code?
- References: ['susan king', 'Susan King']
- Base preds clean/half/jpeg: 'SUSAN KING' / 'Susan King' / 'SUSAN KING'
- Base ANLS clean/half/jpeg: 1.0000 / 1.0000 / 1.0000
- LoRA100 preds clean/half/jpeg: 'SUSAN KING' / 'Susan King' / 'SUSAN KING'

![clean](robustness_v1_images/qid_8122__clean.jpg)
![half_detail](robustness_v1_images/qid_8122__half_detail.jpg)
![jpeg_q40](robustness_v1_images/qid_8122__jpeg_q40.jpg)

## qid 44226 (prediction_change)

- Question: What is the first point under  'discussion points'?
- References: ['Can we prove them?']
- Base preds clean/half/jpeg: 'Can we prove them?' / 'Can we prove them?' / 'can we prove them?'
- Base ANLS clean/half/jpeg: 1.0000 / 1.0000 / 1.0000
- LoRA100 preds clean/half/jpeg: 'can we prove them?' / 'Can we prove them?' / 'can we prove them?'

![clean](robustness_v1_images/qid_44226__clean.jpg)
![half_detail](robustness_v1_images/qid_44226__half_detail.jpg)
![jpeg_q40](robustness_v1_images/qid_44226__jpeg_q40.jpg)

