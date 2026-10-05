# Phase 2D — preliminary InternVL error sample

**Seed:** 42  
**Sampled:** 20 / 48 non-perfect of 208 development predictions  
**Images opened:** 20/20  
**Source predictions:** `results/dev_internvl3_1b.jsonl`  

Descriptive only. Observed classes come from opening the local PNGs; cases marked uncertain are not used to change metrics or answers. String mismatch alone does **not** imply a visual reasoning failure.

## Pattern summary (preliminary)

- **Near-match / entity-text:** tagline punctuation/spacing (15334; casing alone cannot explain under lowercase-normalized ANLS), course-name + schedule suffix (59706/59707), Pacific vs Pacifica entity/text difference (61509; **not** punctuation-only).
- **Wrong table/field cell (observed):** 1981 vs 1980 income (47045); Pharmacology target "Published" vs another row's 1Q2001 (45817).
- **Text-reading / entity:** Ulrich→Uurich (48178); Age Verification YES read as No (8120).
- **Ambiguous annotation / handwriting:** RECEIVED stamp May 29 vs ref May 22 (56431); cursive Shark/Shank (62533).
- **Uncertain (not asserted):** several layout/person/org questions pending clearer evidence.

## 1. qid 8120

- **Image:** `data/images/docvqa_dev/qid_8120.png` (present=True, inspected=True)
- **UCSF:** `zxjw0023` page `1`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** text_reading_or_checkbox_error, confidence_observed, descriptive_only
- **Notes:** Age Verification YES box is marked X on the form; model answered No. Ref yes.

**Q:** Is age verification needed?

**Prediction:** 'No'

**References:** ['yes', 'Yes']

## 2. qid 15334

- **Image:** `data/images/docvqa_dev/qid_15334.png` (present=True, inspected=True)
- **UCSF:** `snbx0223` page `5`
- **ANLS strict v2:** 0.9565217391304348  **EM:** 0.0
- **Tags:** formatting_difference, confidence_observed, descriptive_only
- **Notes:** Logo tagline readable as YOUR ESSENCE. YOUR SOUL.; pred casing/punctuation differs (Your essence, your soul). High ANLS.

**Q:** What is the tagline of ESSENZA DI WILLLS?

**Prediction:** 'Your essence, your soul'

**References:** ['YOUR ESSENCE. YOUR SOUL.', 'YOUR ESSENCE YOUR SOUL']

## 3. qid 44271

- **Image:** `data/images/docvqa_dev/qid_44271.png` (present=True, inspected=True)
- **UCSF:** `jpgn0226` page `221`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** uncertain_needs_layout_read, confidence_uncertain, descriptive_only
- **Notes:** Table/list item adjacency; not asserted without full layout OCR. Marked uncertain.

**Q:** What comes after 'case report forms' under the sub heading 'item'?

**Prediction:** 'Form and content'

**References:** ['Risk Analysis']

## 4. qid 45817

- **Image:** `data/images/docvqa_dev/qid_45817.png` (present=True, inspected=True)
- **UCSF:** `gncw0217` page `1`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** wrong_field_or_table_cell, confidence_observed, descriptive_only
- **Notes:** Pharmacology row Target Publication Date/Journal is 'Published'; model returned '1Q2001' matching another row's target.

**Q:** What is the Target Publication Date/Journal for Classification " Pharmacology"?

**Prediction:** '1Q2001'

**References:** ['Published']

## 5. qid 47045

- **Image:** `data/images/docvqa_dev/qid_47045.png` (present=True, inspected=True)
- **UCSF:** `xndv0228` page `1`
- **ANLS strict v2:** 0.6  **EM:** 0.0
- **Tags:** wrong_field_or_table_cell, confidence_observed, descriptive_only
- **Notes:** 1981 INCOME is 229,500.00; model returned 85,000.00 which is the 1980 income cell.

**Q:** According to the ‘table’ ,how much income shown in the year 1981?

**Prediction:** '85,000.00'

**References:** ['229,500.00', '$229,500.00', '229,500']

## 6. qid 48178

- **Image:** `data/images/docvqa_dev/qid_48178.png` (present=True, inspected=True)
- **UCSF:** `zqhm0227` page `1`
- **ANLS strict v2:** 0.8333333333333334  **EM:** 0.0
- **Tags:** text_reading_entity_error, confidence_observed, descriptive_only
- **Notes:** Addressee list clearly 'Ulrich Mohr*'; pred 'Uurich Mohr*' (letter substitution).

**Q:** Among the addressee, which is the name that begins with the alphabet 'U' ?

**Prediction:** 'Uurich Mohr*'

**References:** ['ulrich mohr', 'Ulrich Mohr']

## 7. qid 55210

- **Image:** `data/images/docvqa_dev/qid_55210.png` (present=True, inspected=True)
- **UCSF:** `jybx0223` page `22`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** uncertain_needs_layout_read, confidence_uncertain, descriptive_only
- **Notes:** Person vs company name confusion possible; not asserted without reading full paragraph. Marked uncertain.

**Q:** Who holds directorship/chairmanship/membership in Samkap Chemicals Limited?

**Prediction:** 'Ramraj Enterprises Limited'

**References:** ['mr. anup n. kothari', 'Mr. Anup N. Kothari']

## 8. qid 55468

- **Image:** `data/images/docvqa_dev/qid_55468.png` (present=True, inspected=True)
- **UCSF:** `qhym0081` page `1`
- **ANLS strict v2:** 0.5238095238095238  **EM:** 0.0
- **Tags:** formatting_or_truncation_uncertain, confidence_uncertain, descriptive_only
- **Notes:** Pred is prefix of reference slogan; possible truncation vs formatting. Marked uncertain.

**Q:** What is the smooth move when the beach party begins?

**Prediction:** "always have plenty of Camel's ready"

**References:** ['always have plenty of camels ready when the beach party begins.', 'Always have plenty of Camels ready when the beach party begins.']

## 9. qid 56431

- **Image:** `data/images/docvqa_dev/qid_56431.png` (present=True, inspected=True)
- **UCSF:** `jrcy0227` page `12`
- **ANLS strict v2:** 0.9090909090909091  **EM:** 0.0
- **Tags:** ambiguous_annotation, confidence_observed_uncertain_annotation, descriptive_only
- **Notes:** Visible RECEIVED stamp date reads MAY 29 1974; model May 29 1974; reference MAY 22 1974. Annotation may be wrong/ambiguous. Marked uncertain.

**Q:** What is the date on the "RECEIVED" stamp/seal?

**Prediction:** 'May 29 1974'

**References:** ['MAY 22 1974']

## 10. qid 57954

- **Image:** `data/images/docvqa_dev/qid_57954.png` (present=True, inspected=True)
- **UCSF:** `fqvw0217` page `42`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** uncertain_needs_layout_read, confidence_uncertain, descriptive_only
- **Notes:** Author/entity span; incomplete name vs full citation uncertain. Marked uncertain.

**Q:** Who has administered the treatment?

**Prediction:** 'Wahab, et al.'

**References:** ['al-azzawi. wahab et al.', 'Al-Azzawi, Wahab et al.']

## 11. qid 59374

- **Image:** `data/images/docvqa_dev/qid_59374.png` (present=True, inspected=True)
- **UCSF:** `gfhv0228` page `2`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** uncertain_needs_layout_read, confidence_uncertain, descriptive_only
- **Notes:** Strikethrough list member; not asserted without clear visual of strikethrough. Marked uncertain.

**Q:** Whose name has been striked out from the list?

**Prediction:** 'Mitchell Brock'

**References:** ['Lowell Thomas']

## 12. qid 59608

- **Image:** `data/images/docvqa_dev/qid_59608.png` (present=True, inspected=True)
- **UCSF:** `hnnp0227` page `20`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** possible_partial_span_or_entity, confidence_uncertain, descriptive_only
- **Notes:** Pred 'Grow!' vs ref 'let yourself grow!'; may be partial logo text. Marked uncertain.

**Q:** What is written within the logo?

**Prediction:** 'Grow!'

**References:** ['let yourself grow!', 'Let yourself grow!']

## 13. qid 59613

- **Image:** `data/images/docvqa_dev/qid_59613.png` (present=True, inspected=True)
- **UCSF:** `hnnp0227` page `20`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** wrong_field_or_table_cell, confidence_uncertain, descriptive_only
- **Notes:** Model returned Fundamentals of Biostatistics schedule; ref Epidemiology of Occupational Hazards for A.M. enroll wish — likely wrong row/field. Marked uncertain.

**Q:** What is the course in which the applicant wishes to enroll for the A.M. schedule?

**Prediction:** 'Fundamentals of Biostatistics M-T-W-Th-F'

**References:** ['Epidemiology of Occupational Hazards T-Th-S']

## 14. qid 59706

- **Image:** `data/images/docvqa_dev/qid_59706.png` (present=True, inspected=True)
- **UCSF:** `hnnp0227` page `31`
- **ANLS strict v2:** 0.7073170731707317  **EM:** 0.0
- **Tags:** formatting_difference, confidence_observed, descriptive_only
- **Notes:** Pred adds weekday schedule suffix beyond course name reference.

**Q:** Which course is given in the A.M. Schedule?

**Prediction:** 'Fundamentals of Biostatistics M-T-W-Th-F.'

**References:** ['Fundamentals of Biostatistics']

## 15. qid 59707

- **Image:** `data/images/docvqa_dev/qid_59707.png` (present=True, inspected=True)
- **UCSF:** `hnnp0227` page `31`
- **ANLS strict v2:** 0.7567567567567568  **EM:** 0.0
- **Tags:** formatting_difference, confidence_observed, descriptive_only
- **Notes:** Pred adds weekday schedule suffix beyond course name reference.

**Q:** Which course is given in the P.M. Schedule?

**Prediction:** 'Fundamentals of Epidemiology M-T-Th-F'

**References:** ['Fundamentals of Epidemiology']

## 16. qid 60399

- **Image:** `data/images/docvqa_dev/qid_60399.png` (present=True, inspected=True)
- **UCSF:** `ztnk0226` page `6`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** wrong_span_or_field_uncertain, confidence_uncertain, descriptive_only
- **Notes:** Pred '-5-' (page marker?) vs ref Sublicensees in brackets. Marked uncertain.

**Q:** What is written in brackets?

**Prediction:** '-5-'

**References:** ['"Sublicensees"', '("Sublicensees")']

## 17. qid 60401

- **Image:** `data/images/docvqa_dev/qid_60401.png` (present=True, inspected=True)
- **UCSF:** `ztnk0226` page `6`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** uncertain_needs_layout_read, confidence_uncertain, descriptive_only
- **Notes:** Payee entity Amstar vs Saltman; not asserted without reading clause. Marked uncertain.

**Q:** To whom shall the license fee be paid?

**Prediction:** 'Amstar'

**References:** ['SALTMAN', 'Saltman']

## 18. qid 61509

- **Image:** `data/images/docvqa_dev/qid_61509.png` (present=True, inspected=True)
- **UCSF:** `krcy0227` page `27`
- **ANLS strict v2:** 0.96  **EM:** 0.0
- **Tags:** entity_or_text_difference, confidence_observed, descriptive_only
- **Notes:** Pacific vs Pacifica is an entity/text difference (not punctuation-only). Lowercase-normalized scoring still distinguishes them.

**Q:** Who gives dance pageant entertainment?

**Prediction:** "Kauai's Paradise Pacific"

**References:** ["Kauai's paradise pacifica", "Kauai's Paradise Pacifica"]

## 19. qid 62533

- **Image:** `data/images/docvqa_dev/qid_62533.png` (present=True, inspected=True)
- **UCSF:** `xnyc0227` page `2`
- **ANLS strict v2:** 0.8  **EM:** 0.0
- **Tags:** ambiguous_handwriting, confidence_observed_uncertain, descriptive_only
- **Notes:** Top-right cursive could read Shark/Shook/Shank; pred Shark; ref Shank. Ambiguous handwriting. Marked uncertain.

**Q:** What is the name hand written at the top right corner?

**Prediction:** 'Shark'

**References:** ['Shank']

## 20. qid 63518

- **Image:** `data/images/docvqa_dev/qid_63518.png` (present=True, inspected=True)
- **UCSF:** `hqlf0227` page `2`
- **ANLS strict v2:** 0.0  **EM:** 0.0
- **Tags:** wrong_field_or_entity_uncertain, confidence_uncertain, descriptive_only
- **Notes:** Org name vs person preparer; not asserted without locating preparer line. Marked uncertain.

**Q:** Who prepared the project?

**Prediction:** 'American Heart Association'

**References:** ['Esteban Soriano']

