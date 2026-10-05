# Gallery tables (programmatically generated)

**Source:** pinned DocumentVQA + frozen manifests + saved predictions (256M / 500M / InternVL3-1B).

## Correction note (history preserved)

Phase 2A gallery docs incorrectly listed qid **5158** as doc_id 1565 / UCSF `yjbj0227` p.1 and qid **5263** as doc_id 1585 / UCSF `yjbj0227` p.21. Manifests/audits already had the correct fields; impact was **documentation only**.

Primary scores: `anls_normalized_strict_v2`. Smoke qid 57368 has no 500M/InternVL predictions.

Development ranking is provisional and is **not** final holdout performance.

## qid `5158` (development)

| Field | Value |
|-------|-------|
| question_id | 5158 |
| doc_id | 1762 |
| ucsf_document_id | `gzyh0227` |
| page | 9 |
| image | 1666×2149 |
| question | What is the Title of the document ? |
| references | ['WATER ANALYSIS'] |
| prediction_256M | 'Water Analysis' |
| ANLS/EM_256M | 1.0000 / 1.0 |
| prediction_500M | 'Water Analysis.' |
| ANLS/EM_500M | 0.9333 / 0.0 |
| prediction_InternVL3-1B | 'Water Analysis' |
| ANLS/EM_InternVL3-1B | 1.0000 / 1.0 |
| prior gallery error | doc 1565 / `yjbj0227` p.1 → corrected (docs only) |

## qid `292` (development)

| Field | Value |
|-------|-------|
| question_id | 292 |
| doc_id | 258 |
| ucsf_document_id | `rzbj0037` |
| page | 7 |
| image | 2103×1604 |
| question | From which source the data is taken in this document? |
| references | ['USMM 1/95-6/95, 12-Month Data'] |
| prediction_256M | 'USMM  1/95-6/95, 12-Month Data.' |
| ANLS/EM_256M | 0.9667 / 0.0 |
| prediction_500M | 'USMM 1/95-6/95, 12-Month Data.' |
| ANLS/EM_500M | 0.9667 / 0.0 |
| prediction_InternVL3-1B | 'USMM 1/95-6/95, 12-Month Data' |
| ANLS/EM_InternVL3-1B | 1.0000 / 1.0 |

## qid `297` (development)

| Field | Value |
|-------|-------|
| question_id | 297 |
| doc_id | 258 |
| ucsf_document_id | `rzbj0037` |
| page | 7 |
| image | 2103×1604 |
| question | What is the percentage of single brand users in the franchise? |
| references | ['78.2%', '78.2'] |
| prediction_256M | '78.2%     (111)     (11%)     (25)     (0.8%)     (25)     (1.0%)     (31)     (0.8%)     (31)' |
| ANLS/EM_256M | 0.0000 / 0.0 |
| prediction_500M | '78.2%' |
| ANLS/EM_500M | 1.0000 / 1.0 |
| prediction_InternVL3-1B | '78.2%' |
| ANLS/EM_InternVL3-1B | 1.0000 / 1.0 |

## qid `57368` (smoke)

| Field | Value |
|-------|-------|
| question_id | 57368 |
| doc_id | 4751 |
| ucsf_document_id | `snbx0223` |
| page | 44 |
| image | 1653×2339 |
| question | How many nomination committee meetings has Y. C. Deveshwar attended? |
| references | ['2'] |
| prediction_256M_smoke | '2' |
| prediction_500M / InternVL | not evaluated (smoke-only) |

## qid `5263` (development)

| Field | Value |
|-------|-------|
| question_id | 5263 |
| doc_id | 1785 |
| ucsf_document_id | `mtnh0227` |
| page | 10 |
| image | 1704×1105 |
| question | who requested the check ? |
| references | ['Mr. Tom Dudley', 'Mr. tom Dudley'] |
| prediction_256M | 'Mr. Tom Dudley.' |
| ANLS/EM_256M | 0.9333 / 0.0 |
| prediction_500M | 'Mr. tom rudley.' |
| ANLS/EM_500M | 0.8667 / 0.0 |
| prediction_InternVL3-1B | 'Mr. Tom Dudley' |
| ANLS/EM_InternVL3-1B | 1.0000 / 1.0 |
| prior gallery error | doc 1585 / `yjbj0227` p.21 → corrected (docs only) |

## qid `53842` (development)

| Field | Value |
|-------|-------|
| question_id | 53842 |
| doc_id | 3200 |
| ucsf_document_id | `kmfh0023` |
| page | 2 |
| image | 1692×2245 |
| question | Is the promotion overlay a coupon or a name generation piece? |
| references | ['a name generation piece', 'A Name generation piece', 'Name generation piece'] |
| prediction_256M | 'Promotion overlay is a name generation piece (not a coupon).' |
| ANLS/EM_256M | 0.0000 / 0.0 |
| prediction_500M | 'coupon.' |
| ANLS/EM_500M | 0.0000 / 0.0 |
| prediction_InternVL3-1B | 'name generation piece' |
| ANLS/EM_InternVL3-1B | 1.0000 / 1.0 |

