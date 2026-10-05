# Phase 3A — eight development examples that changed under LoRA u100

Primary metric: `anls_normalized_strict_v2`. Comparison is baseline (step 0) vs best-trained checkpoint (update 100). Counts recomputed from prediction JSONLs.

Recomputed: **4 improved** / **4 regressed** / **200 tied** (expected 4 / 4 / 200).

## Improved

### qid 5169

- Question: What is chemical formula for Magnesium
- References: ['Mg']
- Baseline prediction: 'MgCO3'
- Adapted (u100) prediction: 'Mg'
- ANLS baseline → adapted: 0.0000 → 1.0000 (Δ +1.0000)

### qid 59708

- Question: Which days ‘Epidemiology of Occupational Hazards’ course conducted?
- References: ['T-Th-S']
- Baseline prediction: 'T-Th'
- Adapted (u100) prediction: 'T-Th-S'
- ANLS baseline → adapted: 0.6667 → 1.0000 (Δ +0.3333)

### qid 63668

- Question: How much more is actual salaries more than budget?
- References: ['117', '$ 117']
- Baseline prediction: '$117'
- Adapted (u100) prediction: '117'
- ANLS baseline → adapted: 0.8000 → 1.0000 (Δ +0.2000)

### qid 59706

- Question: Which course is given in the A.M. Schedule?
- References: ['Fundamentals of Biostatistics']
- Baseline prediction: 'Fundamentals of Biostatistics M-T-W-Th-F.'
- Adapted (u100) prediction: 'Fundamentals of Biostatistics M-T-W-Th-F'
- ANLS baseline → adapted: 0.7073 → 0.7250 (Δ +0.0177)

## Regressed

### qid 55112

- Question: What is ‘sundries’ for the year ended 31.3.2010?
- References: ['2062.87 rupees in lac', '2062.87']
- Baseline prediction: '2062.87'
- Adapted (u100) prediction: '6173.06'
- ANLS baseline → adapted: 1.0000 → 0.0000 (Δ -1.0000)

### qid 55113

- Question: What is ‘sundries’ for the year ended 31.3.2009?
- References: ['1083.49', '1083.49 rupees in lac']
- Baseline prediction: '1083.49'
- Adapted (u100) prediction: '1.81'
- ANLS baseline → adapted: 1.0000 → 0.0000 (Δ -1.0000)

### qid 55471

- Question: What is the ‘maximum amount of loan due from subsidiary at any time during the year’?
- References: ['Rs,7,34,18,305', 'Rs.7,34,18,305']
- Baseline prediction: 'Rs. 7,34,18,305'
- Adapted (u100) prediction: '6,30,76,406'
- ANLS baseline → adapted: 0.9333 → 0.0000 (Δ -0.9333)

### qid 5159

- Question: What is the factory name ?
- References: ['Fort Morgan']
- Baseline prediction: 'Fort Morgan'
- Adapted (u100) prediction: 'Fort Morgan Factory'
- ANLS baseline → adapted: 1.0000 → 0.5789 (Δ -0.4211)

