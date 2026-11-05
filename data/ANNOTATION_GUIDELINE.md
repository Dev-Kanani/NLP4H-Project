# Annotation Guideline — Abdominal USG Findings (v2, reports US_SYNTH_051 onward)

Corpus: abdominal (+ pelvis) ultrasound reports from corporate preventive health checkups, working-age
adults (22–62 y), India. Each report is annotated as a list of **finding rows**.

## 1. Row fields

| field | values |
|---|---|
| `organ` | exactly one of `Liver`, `Gallbladder`, `Pancreas`, `Spleen`, `Kidneys`, `Pelvis` |
| `finding` | a name from the vocabulary in §3 (exact spelling) |
| `grade` | `Grade 1`, `Grade 2`, `Grade 3` **only for Steatosis**; every other row `N/A` |
| `size_mm` | number in **millimetres** (convert cm × 10), or `null` — see §5 |
| `status` | `Affirmed`, `Negated`, `Possible` — see §4 |
| `evidence_text` | a **verbatim, contiguous substring** of the report text (copy-paste exact, including case and punctuation) that shows the finding and its status cue |

Codes (RadLex / ICD-10-CM) are filled in automatically from the finding name; do not add them.

## 2. What to annotate

1. Annotate a finding when the report **explicitly names it** (or an unambiguous synonym / abbreviation),
   whether it is present (Affirmed), absent (Negated) or uncertain (Possible).
   - "No calculus or hydronephrosis" → two rows (Calculus Negated, Hydronephrosis Negated).
   - "PCS not dilated" / "no hydro" / "no HDN" → Hydronephrosis Negated.
   - "No SOL" → Focal Lesion Negated.
2. **Do not** annotate purely normal *attribute* descriptions that do not name a finding: "normal in size",
   "normal echotexture", "CMD maintained", "CBD 4 mm", "wall regular", "smooth outline", organ measurements.
   ("Liver normal in size" is **not** Hepatomegaly Negated; "No hepatomegaly" **is**.)
3. **Unremarkable** (organ-level normality): annotate `Unremarkable` / Affirmed when an organ is declared normal
   *as a whole*: "unremarkable", "NAD", "no abnormality detected", "normal", "appears normal",
   "normal sonographic appearance", "WNL" used as the whole statement about the organ ("Spleen: WNL").
   Not for partial statements ("WNL size", "normal in size and echotexture").
   Never annotate Unremarkable for an organ that also has an Affirmed or Possible abnormality.
   Combined statements ("Pancreas and spleen appear normal") → one row per organ.
4. **Unremarkable Study**: when the report concludes the *whole examination* is normal
   ("Normal study", "No sonographic abnormality detected", "Impression: NAD", "unremarkable abdominal
   ultrasound"), add one row `organ = Pelvis`, `finding = Unremarkable Study`, Affirmed (legacy convention:
   study-level rows are filed under Pelvis). Not when qualified ("otherwise normal", "no other abnormality").
5. **Impression / Opinion / Conclusion** sections usually repeat the body: do **not** annotate restatements.
   Annotate an impression statement only if it adds a finding that the body never mentions (then file it
   under its organ). Exception: Unremarkable Study (rule 4) usually lives in the impression.
6. **One row per mention per sub-section.** If the same finding + status appears twice in one organ's passage,
   annotate once. But separate sub-sections are separate rows: a "RIGHT KIDNEY:" line and a "LEFT KIDNEY:"
   line that each say "no hydronephrosis" → two Hydronephrosis Negated rows (organ Kidneys).
   "Both kidneys: no calculus" → one row.
7. **Distinct lesions** described separately with their own size → separate rows (right renal cyst 11 mm and
   left renal cyst 16 mm → two Cortical Cyst rows). A group described together ("multiple calculi, largest
   7 mm") → one row with the largest size.
8. Clinical advice ("Suggested LFT correlation", "Adv: CECT") is not annotated.
9. Identifiers (patient name, dates, MRN/UHID, phone, referring doctor, signing radiologist) are not findings;
   list them separately as PHI (writers only, see task instructions).

## 3. Vocabulary (organ → finding names)

**Liver:** `Steatosis` (fatty liver / fatty infiltration / hepatic steatosis / bright liver / increased
echogenicity attributed to fat), `Hepatomegaly`, `Focal Lesion` (SOL, mass, nodule, "focal lesion"),
`Hemangioma`, `Hepatic Cyst`, `Intrahepatic Biliary Dilatation` (IHBR / intrahepatic ducts), `Biliary Dilatation`
(biliary dilatation without "intrahepatic"/"CBD" qualifier), `Unremarkable`

**Gallbladder** (includes the CBD): `Calculus` (stones, cholelithiasis), `Sludge`, `Polyp`, `Wall Thickening`,
`Pericholecystic Fluid`, `Sonographic Murphy Sign`, `Adenomyomatosis`, `Cholecystectomy Status`
(gallbladder surgically removed — Affirmed), `CBD Dilatation`, `Unremarkable`

**Pancreas:** `Focal Lesion` (when worded lesion/mass/SOL), `Focal Abnormality` (when worded "abnormality"),
`Calcification`, `Ductal Dilatation` (MPD), `Surrounding Collection` (peripancreatic collection/fluid),
`Bowel Gas Obscuration` (Affirmed when gas obscures/limits the organ), `Unremarkable`

**Spleen:** `Splenomegaly`, `Focal Lesion`, `Accessory Spleen` (splenule), `Unremarkable`

**Kidneys:** `Calculus` (stone, nephrolithiasis, calculi, echogenic shadowing focus called a calculus),
`Hydronephrosis` (incl. hydroureteronephrosis, PCS dilatation, HDN), `Cortical Cyst` (simple renal cyst),
`Septation`, `Solid Component`, `Angiomyolipoma`, `Focal Lesion`, `Unremarkable`

**Pelvis** (urinary bladder, prostate, uterus, ovaries): `Free Fluid` (incl. ascites, fluid in POD/Morison's
pouch is filed here too), `Pelvic Mass`, `Prostatomegaly`, `Uterine Fibroid` (fibroid / leiomyoma / myoma),
`Adenomyosis`, `Ovarian Cyst`, `Bladder Wall Thickening`, `Unremarkable`, `Unremarkable Study`

`Bowel Gas Obscuration` may be used under any organ whose assessment is limited by gas.
If something genuinely does not fit any name, use a short Title Case name and flag it in your notes.

## 4. Status

- **Affirmed**: stated present, including "consistent with", "suggestive of", "s/o", "in keeping with",
  "compatible with", "features of", "physiological" (physiological fluid is still fluid → Affirmed).
- **Negated**: "no", "not seen", "absent", "without", "nil", "ruled out", "negative" (Murphy sign negative),
  "not dilated" (ducts / PCS), "not thickened".
- **Possible**: "?", "possible", "probable", "likely", "suspicious for", "may represent", "could be",
  "cannot be excluded / ruled out", "doubtful", "query", "equivocal", "borderline … ? early".
  If a lesion is definitely present but its *nature* is uncertain ("4 cm ovarian cyst, possibly haemorrhagic")
  the cyst is **Affirmed**; if the existence itself is uncertain ("? tiny calculus") it is **Possible**.
  "Likely angiomyolipoma" for a seen lesion → `Angiomyolipoma` Possible.

## 5. Attributes

- **Grade (Steatosis only):** explicit grade wins (Grade I/II/III, grade-1, G2). Otherwise mild → Grade 1,
  moderate → Grade 2, severe/marked → Grade 3. No severity information → `N/A`.
- **size_mm:** only for lesions/stones/cysts/polyps/fibroids/masses/nodules. Largest dimension, in mm
  (1.2 × 0.9 cm → 12). Never for organ dimensions (liver span, spleen length, kidney size, CBD calibre,
  wall thickness, prostate volume). Negated rows → `null`.

## 6. Clarifications (v2.1)

1. **Unremarkable granularity.** One `Unremarkable` row per organ *per sub-section or separate statement*.
   Separate lines/sentences ("RIGHT KIDNEY: normal" … "LEFT KIDNEY: normal"; "UB: normal. Prostate: normal")
   → separate rows. One combined statement ("UB, Ut, B/L Ov - WNL"; "Liver, GB, Spl - NAD") → one row per
   organ value it covers (here: one Pelvis row; one Liver, one Gallbladder, one Spleen row).
2. Rule 3 still applies per organ value: if the organ (e.g. Pelvis) has any Affirmed/Possible abnormality
   (fibroid, ovarian cyst, prostatomegaly, free fluid…), add **no** Unremarkable row for that organ.
3. "Rest of abdomen NAD", "Otherwise NAD/normal", "Rest NAD" → **nothing** (no Unremarkable Study, and do not
   expand into per-organ rows).
4. Age-appropriate or technical states with no vocabulary name are **not annotated**: atrophic/postmenopausal
   uterus, ovaries not visualised, contracted/non-fasting gallbladder, bulky uterus (without a named lesion),
   partial-organ statements ("head and body normal").
5. "Biliary radicles / IHBRs not dilated" → `Intrahepatic Biliary Dilatation` Negated (radicles are intrahepatic).
   "IHBRs normal" (no explicit negation of dilatation) → not annotated.
6. "No back-pressure changes" / "PCS not dilated" → `Hydronephrosis` Negated.
7. CBD: "CBD not dilated" → `CBD Dilatation` Negated; a bare calibre ("CBD 4 mm") → nothing;
   "no calculus in CBD" → `Calculus` Negated under Gallbladder (separate row from the GB lumen statement).
8. Physiological / non-fasting explanations make a *seen* finding Affirmed; a finding whose existence is
   called "apparent" or "likely physiological rather than true …" is `Possible`.

## 7. Ontology grounding (applied automatically by finding name)

RadLex is assigned by concept regardless of status (legacy convention); ICD-10-CM only to **Affirmed** rows.

| organ / finding | RadLex | ICD-10-CM (Affirmed) |
|---|---|---|
| Liver / Steatosis | RID39103 | K76.0 |
| Liver / Hepatomegaly | – | R16.0 |
| Liver / Focal Lesion | – | R93.2 |
| Liver / Hemangioma | – | D18.03 |
| Liver / Hepatic Cyst | – | K76.89 |
| Gallbladder / Calculus | RID5770 | K80.20 |
| Gallbladder / Polyp, Adenomyomatosis | – | K82.8 |
| Gallbladder / Wall Thickening | – | R93.2 |
| Gallbladder / Cholecystectomy Status | – | Z90.49 |
| Gallbladder / CBD Dilatation | – | K83.8 |
| Pancreas / Calcification, Ductal Dilatation | – | K86.89 |
| Spleen / Splenomegaly | – | R16.1 |
| Spleen / Accessory Spleen | – | Q89.09 |
| Kidneys / Calculus | – | N20.0 |
| Kidneys / Hydronephrosis | – | N13.30 |
| Kidneys / Cortical Cyst | RID3954 | N28.1 |
| Kidneys / Angiomyolipoma | – | D30.00 |
| Pelvis / Prostatomegaly | – | N40.0 |
| Pelvis / Uterine Fibroid | – | D25.9 |
| Pelvis / Ovarian Cyst | – | N83.20 (category level; refine with laterality) |
| Pelvis / Bladder Wall Thickening | – | N32.89 |
| any / Unremarkable | RID28473 | – |
| Pelvis / Unremarkable Study | RID28473 | Z00.00 |

Codes in the first two rows of each organ and the four RadLex IDs come from the original v1 guideline; the rest
were added for v2 and **should be verified** (ICD-10-CM browser, RSNA RadLex) before publication.

## 8. Differences from the v1 annotations (reports 001–050, left unchanged)

| topic | v1 (001–050) | v2 (051–100) |
|---|---|---|
| Organ-level "Unremarkable" | Inconsistent: same wording labelled in some reports, not others; telegraphic "WNL" mostly not labelled; in fully normal narrative reports every organ with normal descriptors labelled | Explicit whole-organ statements only, including "WNL"/"NAD"/"appears normal" (§2.3, §6.1) |
| Laterality | Occasionally a suffix in the finding name (`Cortical Cyst - Right`, `Calculus - Left`) | Never in the finding name |
| Evidence spans | 577/614 verbatim; the rest lower-cased or paraphrased | 616/616 verbatim substrings |
| Pancreas "unremarkable visualized portion" | `Unremarkable Visualized Portion` (once) | `Unremarkable` + separate `Bowel Gas Obscuration` |
| Kidney "no definite shadowing calculus" | `Definite Shadowing Calculus` | `Calculus` Negated |
| Vocabulary | ~30 finding names | v1 names + CBD, sludge, polyp, adenomyomatosis, cholecystectomy, hemangioma, hepatic cyst, accessory spleen, angiomyolipoma, pancreatic calcification/duct, prostate, uterus, ovary, bladder findings |

If the team wants one consistent gold standard for all 100 reports, re-annotate 001–050 under v2.1.
