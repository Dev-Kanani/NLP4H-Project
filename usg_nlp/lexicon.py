"""Rule-baseline gazetteer: regex patterns used ONLY by the rule-based extractor.

Vocabulary, codes, pathological flags and triage policy live in
``resources/ontology.json`` (see :mod:`usg_nlp.ontology`), shared with the LLM
extractor.  This module holds the surface patterns that the *baseline* uses to
find those concepts in text - the part that does not generalise to unseen
phrasing and that the LLM extractor replaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ontology import load_ontology
from .schemas import Organ

# --------------------------------------------------------------------------- targets


@dataclass(frozen=True)
class TargetConcept:
    """A lexical target the extractor searches for inside one or more organ sections."""

    finding: str  # ontology finding name written to the record
    concept: str  # kept for readability; the ontology's concept is authoritative
    organs: tuple[Organ, ...]
    patterns: tuple[str, ...]
    compiled: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compiled", tuple(re.compile(p, re.I) for p in self.patterns))

    @property
    def pathological(self) -> bool:
        return load_ontology().is_pathological(self.organs[0].value, self.finding)


L, GB, P, S, K, PV = Organ.LIVER, Organ.GALLBLADDER, Organ.PANCREAS, Organ.SPLEEN, Organ.KIDNEYS, Organ.PELVIS

TARGETS: tuple[TargetConcept, ...] = (
    # ---------------------------------------------------------------- liver
    TargetConcept(
        "Steatosis", "Steatosis", (L,),
        (
            r"steatosis(?:\s+hepatis)?", r"fatty\s+(?:infiltration|liver|change|changes)",
            r"\b(?:MASLD|NAFLD|MAFLD)\b", r"steatotic\s+liver",
            # descriptive-only phrasing ("diffuse increase in hepatic parenchymal echogenicity")
            r"(?:diffuse(?:ly)?\s+)?(?:increase(?:d)?|raised)\s+(?:in\s+)?(?:diffuse\s+)?(?:hepatic\s+|liver\s+|parenchymal\s+)*echogenicity",
            r"bright\s+liver", r"liver\s+is\s+(?:diffusely\s+)?(?:hyperechoic|bright)",
        ),
    ),
    TargetConcept(
        "Hepatomegaly", "Hepatomegaly", (L,),
        (
            r"hepatomegaly", r"(?:mildly|moderately|markedly)?\s*enlarged\s+liver",
            r"liver\s+(?:is\s+)?(?:mildly\s+|moderately\s+|markedly\s+)?enlarged",
        ),
    ),
    TargetConcept(
        "Focal Lesion", "Focal Lesion", (L,),
        (
            r"focal\s+(?:hepatic\s+)?(?:solid\s+or\s+cystic\s+)?(?:lesion|mass)",
            r"space[\s\-]occupying\s+lesion", r"\bSOL\b",
        ),
    ),
    TargetConcept(
        "Hemangioma", "Hemangioma", (L,), (r"ha?emangioma",),
    ),
    TargetConcept(
        "Intrahepatic Biliary Dilatation", "Biliary Dilatation", (L,),
        (
            r"intrahepatic\s+biliary\s+(?:ductal\s+|radicle\s+)?dilatation",
            r"\bIHBR\s+dilatation\b", r"\bIHBRD\b", r"intrahepatic\s+bile\s+ducts?",
        ),
    ),
    TargetConcept(
        "Biliary Dilatation", "Biliary Dilatation", (L,),
        (r"biliary\s+(?:ductal\s+)?dilatation", r"\bCBD\s+dilatation\b"),
    ),
    # ---------------------------------------------------------------- gallbladder
    TargetConcept(
        "Calculus", "Calculus", (GB,),
        (
            r"gall\s?stones?", r"calcul(?:us|i)", r"\bstones?\b", r"cholelithiasis",
            r"echogenic\s+foc(?:us|i)",
        ),
    ),
    TargetConcept(
        "Wall Thickening", "Wall Thickening", (GB,),
        (
            r"wall\s+thickening", r"thickened\s+(?:gallbladder\s+)?wall",
            r"\bwall\b(?=\s+(?:is|was)\s+(?:not\s+thickened|normal\s+in\s+thickness))",
        ),
    ),
    TargetConcept(
        "Pericholecystic Fluid", "Pericholecystic Fluid", (GB,),
        (r"pericholecystic\s+(?:fluid|collection)", r"surrounding\s+fluid"),
    ),
    TargetConcept(
        "Sonographic Murphy Sign", "Sonographic Murphy Sign", (GB,),
        (r"(?:sonographic\s+)?murphy'?s?\s+sign",),
    ),
    TargetConcept("Sludge", "Sludge", (GB,), (r"(?:biliary\s+)?sludge",)),
    TargetConcept("Polyp", "Polyp", (GB,), (r"polyps?",)),
    # ---------------------------------------------------------------- pancreas
    TargetConcept("Focal Abnormality", "Focal Lesion", (P,), (r"focal\s+(?:pancreatic\s+)?abnormality",)),
    TargetConcept("Focal Lesion", "Focal Lesion", (P,), (r"focal\s+(?:pancreatic\s+)?lesion",)),
    TargetConcept(
        "Surrounding Collection", "Surrounding Collection", (P,),
        (r"(?:surrounding|peripancreatic)\s+collection",),
    ),
    TargetConcept(
        "Bowel Gas Obscuration", "Bowel Gas Obscuration", (P,),
        (r"(?:partly|partially|largely)?\s*obscured\s+by\s+(?:overlying\s+)?(?:bowel\s+)?gas",),
    ),
    TargetConcept("Calcification", "Calcification", (P,), (r"calcifications?",)),
    # ---------------------------------------------------------------- spleen
    TargetConcept("Focal Lesion", "Focal Lesion", (S,), (r"focal\s+(?:splenic\s+)?lesion",)),
    TargetConcept(
        "Splenomegaly", "Splenomegaly", (S,),
        (r"splenomegaly", r"spleen\s+(?:is\s+)?(?:mildly\s+|moderately\s+)?enlarged"),
    ),
    # ---------------------------------------------------------------- kidneys
    TargetConcept(
        "Definite Shadowing Calculus", "Definite Shadowing Calculus", (K,),
        (r"definite\s+shadowing\s+(?:calculus|focus)",),
    ),
    TargetConcept(
        "Calculus", "Calculus", (K,),
        (r"(?:renal\s+)?calcul(?:us|i)", r"(?:renal\s+)?stones?", r"nephrolithiasis"),
    ),
    TargetConcept(
        "Hydronephrosis", "Hydronephrosis", (K,),
        (r"hydronephrosis", r"\bhydro\b", r"pelvicalyceal\s+dilatation"),
    ),
    TargetConcept(
        "Cortical Cyst", "Cortical Cyst", (K,),
        (r"(?:simple\s+)?(?:renal\s+)?(?:cortical\s+)?cysts?\b",),
    ),
    TargetConcept("Septation", "Septation", (K,), (r"septa(?:tions?)?\b",)),
    TargetConcept("Solid Component", "Solid Component", (K,), (r"solid\s+components?",)),
    # ---------------------------------------------------------------- pelvis
    TargetConcept("Free Fluid", "Free Fluid", (PV,), (r"free\s+(?:pelvic\s+)?fluid", r"ascites")),
    TargetConcept("Pelvic Mass", "Pelvic Mass", (PV,), (r"(?:pelvic\s+|adnexal\s+)?mass",)),
    TargetConcept("Prostatomegaly", "Prostatomegaly", (PV,), (r"prostatomegaly", r"enlarged\s+prostate")),
    TargetConcept("Uterine Fibroid", "Uterine Fibroid", (PV,), (r"fibroids?", r"leiomyomas?")),
    TargetConcept("Ovarian Cyst", "Ovarian Cyst", (PV,), (r"ovarian\s+cysts?",)),
)

# Positive organ-level normality descriptors (used for "Unremarkable").
# "WNL" alone is deliberately NOT treated as organ-level normality: the annotation guideline
# reads telegraphic "Liver: WNL" / "WNL size" as a size/echotexture statement only.
EXPLICIT_UNREMARKABLE = re.compile(r"\bunremarkable\b|\bNAD\b|no\s+abnormality\s+detected", re.I)
NORMALITY_DESCRIPTORS: dict[Organ, re.Pattern[str]] = {
    L: re.compile(
        r"normal\s+in\s+size|normal\s+size|homogeneous\s+(?:parenchymal\s+)?echotexture|"
        r"normal\s+echotexture|\bunremarkable\b|\bWNL\b", re.I),
    GB: re.compile(
        r"thin[\s,\-]*(?:smooth\s+)?wall(?:ed)?|normally\s+distended|well\s+distended|"
        r"adequately\s+distended|\bunremarkable\b|\bWNL\b", re.I),
    P: re.compile(r"\bunremarkable\b|normal\s+(?:sonographic\s+)?appearance|normal\s+in\s+appearance|\bWNL\b", re.I),
    S: re.compile(r"normal\s+in\s+size|normal\s+size|normal\s+echotexture|\bunremarkable\b|\bWNL\b", re.I),
    K: re.compile(
        r"normal\s+in\s+size|normal\s+size|preserved\s+size|normal\s+(?:cortical\s+)?echogenicity|"
        r"within\s+normal\s+size\s+limits|\bunremarkable\b|\bWNL\b", re.I),
}
UNREMARKABLE_STUDY = re.compile(
    r"(?<!otherwise\s)\b(?:unremarkable|normal)\s+(?:abdominal\s+)?(?:ultrasound|sonograph\w*|US|USG|study|scan|examination)\b"
    r"|\b(?:examination|study|scan)\s+is\s+(?:unremarkable|normal|within\s+normal\s+limits)\b",
    re.I,
)
STUDY_LEVEL_BLOCKERS = re.compile(r"\botherwise\b|\bother\b|\badditional\b|\bremaining\b", re.I)

# --------------------------------------------------------------------------- attributes

GRADE_EXPLICIT = re.compile(r"\bgrade[\s\-]*(?P<g>III|II|I|[123])\b", re.I)
GRADE_ADJECTIVES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:severe|severely|marked|markedly|pronounced)\b", re.I), "Grade 3"),
    (re.compile(r"\b(?:moderate|moderately)\b", re.I), "Grade 2"),
    (re.compile(r"\b(?:mild|mildly|slight|slightly|minimal)\b", re.I), "Grade 1"),
)
ROMAN_TO_GRADE = {"I": "Grade 1", "II": "Grade 2", "III": "Grade 3", "1": "Grade 1", "2": "Grade 2", "3": "Grade 3"}

MEASUREMENT = re.compile(
    r"(?P<a>\d{1,3}(?:\.\d{1,2})?)\s*(?:(?:x|×|by)\s*(?P<b>\d{1,3}(?:\.\d{1,2})?)\s*)?(?P<unit>mm|cm)\b", re.I
)
LOCATION = re.compile(
    r"\b(?P<loc>upper\s+pole|lower\s+pole|mid[\s\-]pole|interpolar(?:\s+region)?|"
    r"renal\s+sinus|cortex|fundus|neck|body|head|tail)\b",
    re.I,
)
LATERALITY = re.compile(r"\b(?P<side>left|right|bilateral(?:ly)?|both)\b", re.I)
