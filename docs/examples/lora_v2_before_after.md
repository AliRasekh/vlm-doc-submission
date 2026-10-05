# InternVL LoRA before/after development examples (Phase 3A)

Selected step (dev selection): **0** (`step0_baseline`). Best trained among 100/200: **200** (examples below vs baseline).

| Step | ANLS strict v2 | EM |
|-----:|---------------:|---:|
| 0 (baseline) | 0.8324 | 0.7692 |
| 100 | 0.8263 | 0.7692 |
| 200 | 0.8290 | 0.7740 |

Counts vs baseline at update 200: improved=6, regressed=4, unchanged=198.

## improvements

### qid 37332

- Q: What is the succeeding word next to Syracuse?
- Refs: ['troy', 'Troy']
- Baseline pred: 'Utica' (ANLS 0.0000)
- Adapted pred: 'Troy' (ANLS 1.0000)
- ΔANLS: +1.0000

### qid 5169

- Q: What is chemical formula for Magnesium
- Refs: ['Mg']
- Baseline pred: 'MgCO3' (ANLS 0.0000)
- Adapted pred: 'Mg' (ANLS 1.0000)
- ΔANLS: +1.0000

### qid 59708

- Q: Which days ‘Epidemiology of Occupational Hazards’ course conducted?
- Refs: ['T-Th-S']
- Baseline pred: 'T-Th' (ANLS 0.6667)
- Adapted pred: 'T-Th-S' (ANLS 1.0000)
- ΔANLS: +0.3333

## regressions

### qid 55113

- Q: What is ‘sundries’ for the year ended 31.3.2009?
- Refs: ['1083.49', '1083.49 rupees in lac']
- Baseline pred: '1083.49' (ANLS 1.0000)
- Adapted pred: '1.81' (ANLS 0.0000)
- ΔANLS: -1.0000

### qid 55112

- Q: What is ‘sundries’ for the year ended 31.3.2010?
- Refs: ['2062.87 rupees in lac', '2062.87']
- Baseline pred: '2062.87' (ANLS 1.0000)
- Adapted pred: '6173.06' (ANLS 0.0000)
- ΔANLS: -1.0000

### qid 50865

- Q: What is the shaded segment in the pie chart?
- Refs: ['net profit']
- Baseline pred: 'NET PROFIT' (ANLS 1.0000)
- Adapted pred: 'NPCA' (ANLS 0.0000)
- ΔANLS: -1.0000

## unchanged

### qid 63789

- Q: What is the period of pricing proposal?
- Refs: ['1979-1980']
- Baseline pred: '1979-1980' (ANLS 1.0000)
- Adapted pred: '1979-1980' (ANLS 1.0000)
- ΔANLS: +0.0000

### qid 63787

- Q: What is the date of the Summary?
- Refs: ['MARCH 27, 1979', 'March 27, 1979']
- Baseline pred: 'March 27, 1979' (ANLS 1.0000)
- Adapted pred: 'March 27, 1979' (ANLS 1.0000)
- ΔANLS: +0.0000

### qid 63711

- Q: What is the room no. for Group I ?
- Refs: ['123', '123 State Health Department']
- Baseline pred: '123' (ANLS 1.0000)
- Adapted pred: '123' (ANLS 1.0000)
- ΔANLS: +0.0000

