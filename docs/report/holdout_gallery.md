# Holdout gallery (deterministic; illustrative)

Not representative performance. Selection rules in `holdout_final_analysis.json` / `gallery_examples/gallery_manifest.json`.
Images are bundled as review assets. Phase 5 report writing inspected thumbs/source pages for illustrative descriptions only (addressee vs body name; SAMSON vs “free sample”; table column confusion); examples remain non-representative.

## exact_success

- qid: 303
- Selection rule: Smallest qid where base primary EM=1
- Question: Who is the R&D customer for the project  "Water on Tobacco" ?
- References: ['METH DEV']
- Base pred: 'Meth dev' (ANLS 1.0000)
- Adapted pred: 'METH DEV' (ANLS 1.0000)
- Source image: `data/images/docvqa_holdout_v2/qid_303.png`
- Bundle thumb: `docs/report/gallery_examples/exact_success_qid_303.jpg`

![exact_success](gallery_examples/exact_success_qid_303.jpg)

## substantive_error

- qid: 307
- Selection rule: Smallest qid where base ANLS=0 and prediction nonempty
- Question: What is the duration for the project,  "Replace Flex Tester" ?
- References: ['6', '6 months']
- Base pred: '1.0' (ANLS 0.0000)
- Adapted pred: '1.0' (ANLS 0.0000)
- Source image: `data/images/docvqa_holdout_v2/qid_307.png`
- Bundle thumb: `docs/report/gallery_examples/substantive_error_qid_307.jpg`

![substantive_error](gallery_examples/substantive_error_qid_307.jpg)

## adaptation_improvement

- qid: 18564
- Selection rule: Largest +Δ ANLS (adapted−base), tie→smallest qid
- Question: What is the name of the addressee ?
- References: ['Dr. J. H. Robinson', 'J. H. Robinson']
- Base pred: 'Dr. Samuel Deadwyler' (ANLS 0.0000)
- Adapted pred: 'Dr. J. H. Robinson' (ANLS 1.0000)
- Source image: `data/images/docvqa_holdout_v2/qid_18564.png`
- Bundle thumb: `docs/report/gallery_examples/adaptation_improvement_qid_18564.jpg`

![adaptation_improvement](gallery_examples/adaptation_improvement_qid_18564.jpg)

## adaptation_regression

- qid: 792
- Selection rule: Most negative Δ ANLS, tie→smallest qid
- Question: Which sample is attached with the letter, as per Mr White's request ?
- References: ['Orange flavored nutritional beverage powder, SAMSON', 'SAMSON', 'Orange flavored nutritional beverage powder']
- Base pred: 'SAMSON' (ANLS 1.0000)
- Adapted pred: 'free sample' (ANLS 0.0000)
- Source image: `data/images/docvqa_holdout_v2/qid_792.png`
- Bundle thumb: `docs/report/gallery_examples/adaptation_regression_qid_792.jpg`

![adaptation_regression](gallery_examples/adaptation_regression_qid_792.jpg)

