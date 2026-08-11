# Phase 2c — Systematic Composite-Material Audit

**Method:** All three source-dataset category maps were extracted from `scripts/preprocess_pipeline.py`
(`GCV2_MAP`, `GC12_MAP`, `TACO_MAP`) and inverted to enumerate every raw subcategory feeding
into each of the 8 fine-grained target classes. Each subcategory is assessed for
composite/mixed-material risk using the same criteria as the Phase 2b paper/cardboard
and Phase 1 textile findings:

> A subcategory is flagged **⚠ COMPOSITE RISK** if its name, conventional use, or known
> physical construction implies a mixed or coated material (wax coating, plastic lining,
> laminate layer, metallic foil, adhesive-bonded layers, synthetic/natural fibre mix,
> multiple bonded substrate types) that would make a single biodegradable/non-biodegradable
> binary label unreliable.

---

## Per-Class Audit

---

### 1. `battery`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `battery` | ✅ Clean — single electrochemical cell |
| gc12_ | `batteries` | ✅ Clean — same |
| taco_ | `Battery` | ✅ Clean — same |

**Verdict: CLEAN.** Batteries are a well-defined, single-category non-biodegradable item.
No composite ambiguity; a battery is never mistaken for biodegradable material.

---

### 2. `glass`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `glass` | ✅ Clean — plain glass |
| gc12_ | `green-glass` | ✅ Clean — coloured glass bottle |
| gc12_ | `brown-glass` | ✅ Clean — coloured glass bottle |
| gc12_ | `white-glass` | ✅ Clean — colourless glass bottle |
| taco_ | `Glass bottle` | ✅ Clean |
| taco_ | `Glass cup` | ✅ Clean |
| taco_ | `Glass jar` | ✅ Clean |
| taco_ | `Broken glass` | ✅ Clean — still pure glass regardless of state |

**Verdict: CLEAN.** All glass subcategories are pure glass. Bottles occasionally carry
paper labels or metal caps, but the object itself is a single material. No composite
splitting needed.

---

### 3. `metal`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `metal` | ✅ Clean — generic metal items |
| gc12_ | `metal` | ✅ Clean |
| taco_ | `Drink can` | ✅ Clean — aluminium |
| taco_ | `Food can` | ✅ Clean — steel/tinplate |
| taco_ | `Aerosol` | ✅ Clean — steel/aluminium canister |
| taco_ | `Metal bottle cap` | ✅ Clean |
| taco_ | `Scrap metal` | ✅ Clean |
| taco_ | `Aluminium foil` | ✅ Clean — pure aluminium |
| taco_ | `Aluminium blister pack` | ⚠ **COMPOSITE RISK** — blister packs bond aluminium foil to a PVC/PET plastic backing; neither layer alone is clean metal or clean plastic |

**Verdict: MOSTLY CLEAN, one flag.**
`Aluminium blister pack` is a classic composite: aluminium push-through layer bonded to a PVC or PET backing sheet. In practice, these are correctly non-biodegradable but the metal label mislabels the polymer component. Since this feeds into `metal` (non-biodegradable) — the same *binary class* as the correct answer — this does **not** cause a Stage 1 error. It is a fine-grained taxonomy concern but **not a Stage 1 accuracy driver.**

---

### 4. `organic`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `biological` | ✅ Clean — food/plant organic matter |
| gc12_ | `biological` | ✅ Clean |
| taco_ | `Food waste` | ✅ Clean — plain organic material |

**Verdict: CLEAN.** All organic subcategories are unambiguously biodegradable single-material
food or plant waste. No composite risk.

---

### 5. `plastic`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `plastic` | ✅ Clean — generic plastic items |
| gc12_ | `plastic` | ✅ Clean |
| taco_ | `Plastic bottle` | ✅ Clean — HDPE/PET |
| taco_ | `Plastic cup` | ✅ Clean — PS/PP |
| taco_ | `Plastic lid` | ✅ Clean |
| taco_ | `Plastic straw` | ✅ Clean |
| taco_ | `Six pack rings` | ✅ Clean — LDPE |
| taco_ | `Styrofoam piece` | ✅ Clean — EPS foam |
| taco_ | `Plastic bag` | ✅ Clean — LDPE/HDPE |
| taco_ | `Plastic film` | ✅ Clean — single-layer film |
| taco_ | `Other plastic` | ⚠ **COMPOSITE RISK (moderate)** — catch-all category; in practice this includes multi-layer flexible packaging (crisp packets, stand-up pouches, retort pouches) that combine metalised polyester, LDPE, and other films bonded together. Also includes plastic-coated paper cups and composite trays. Visually indistinguishable from single-material plastic but physically composite. |

**Verdict: ONE MODERATE FLAG.**
`Other plastic` is a known catch-all in TACO. Many instances are multi-layer laminate
flexible packaging (metalised film snack pouches, retort pouches, composite condiment
sachets). These are correctly non-biodegradable so Stage 1 labelling is *correct*, but
the visual appearance is heterogeneous and overlaps with paper/foil. This is not causing
Stage 1 errors (they're already non-biodegradable) but does create confusion when
Stage 1 sees similar-looking specimens filed under `paper`.

---

### 6. `textile`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `shoes` | ⚠ **COMPOSITE RISK** — shoes combine leather/textile uppers, rubber soles, synthetic foam midsoles, metal eyelets, and synthetic thread. A leather shoe is partially biodegradable (leather) and partially not (rubber sole, synthetic thread). |
| gcv2_ | `clothes` | ⚠ **COMPOSITE RISK** — "clothes" includes 100% natural-fibre garments (cotton, linen, wool — biodegradable) and 100% synthetic garments (polyester, nylon — non-biodegradable) and blends. The binary label non-biodegradable is assigned to the whole class, which is partially incorrect for pure-natural items. |
| gc12_ | `shoes` | ⚠ **COMPOSITE RISK** — same as gcv2_ shoes |
| gc12_ | `clothes` | ⚠ **COMPOSITE RISK** — same as gcv2_ clothes |
| taco_ | `Rope` | ⚠ **COMPOSITE RISK** — rope can be natural fibre (jute, hemp — biodegradable) or synthetic (nylon, polypropylene — non-biodegradable); visually indistinguishable in photos |
| taco_ | `Shoe` | ⚠ **COMPOSITE RISK** — same as above |
| taco_ | `Clothes` | ⚠ **COMPOSITE RISK** — same as above |

**Verdict: HIGHEST AMBIGUITY CLASS — already flagged in Phase 1.**
`textile` has the most pervasive composite problem of all 8 classes. Every subcategory
feeding into it carries composite risk. The entire class label is structurally unstable
as a binary biodegradable/non-biodegradable assignment.

---

### 7. `paper`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `paper` | ⚠ **COMPOSITE RISK (partial)** — catch-all including plain sheets AND magazines (glossy coated stock), receipts (BPA thermal paper), paper cups (wax/plastic-lined) |
| gc12_ | `paper` | ⚠ **COMPOSITE RISK (partial)** — same catch-all issue |
| taco_ | `Normal paper` | ✅ Clean — plain paper |
| taco_ | `Paper bag` | ✅ Clean — kraft paper; some have wax coating but most are plain |
| taco_ | `Magazine paper` | ⚠ **COMPOSITE RISK** — coated with clay/kaolin or polymer binding; not biodegradable in standard composting timelines; often contains ink binders and laminates |
| taco_ | `Wrapping paper` | ⚠ **COMPOSITE RISK** — gift wrapping paper is commonly plastic-coated, metallic-laminated, or glitter-covered; these are non-biodegradable despite the paper substrate |
| taco_ | `Paper cup` | ⚠ **COMPOSITE RISK** — a direct match for the wax/plastic-lined cups confirmed in Phase 2b visual inspection; PE-lined cups are non-compostable |
| taco_ | `Paper straw` | ✅ Clean — typically uncoated food-safe paper |
| taco_ | `Toilet tube` | ✅ Clean — uncoated cardboard tube |

**Verdict: CONFIRMED COMPOSITE PROBLEM — corroborates Phase 2b.**
Multiple TACO subcategories (`Magazine paper`, `Wrapping paper`, `Paper cup`) are explicitly
composite items incorrectly filed under biodegradable `paper`. The gcv2_ and gc12_ catch-all
`paper` categories contain the same spectrum. This is the dominant driver of the 25.49%
paper error rate.

---

### 8. `cardboard`

| Source | Raw Subcategory | Composite Risk |
|---|---|---|
| gcv2_ | `cardboard` | ⚠ **COMPOSITE RISK (partial)** — catch-all includes plain cardboard AND beverage cartons (Tetra Pak: PE + cardboard + aluminium) |
| gc12_ | `cardboard` | ⚠ **COMPOSITE RISK (partial)** — same |
| taco_ | `Corrugated carton` | ✅ Clean — plain corrugated cardboard |
| taco_ | `Egg carton` | ✅ Clean — moulded pulp |
| taco_ | `Meal carton` | ⚠ **COMPOSITE RISK** — fast-food/takeaway meal cartons are often wax-coated or PE-coated for grease resistance; not biodegradable |
| taco_ | `Pizza box` | ⚠ **COMPOSITE RISK (moderate)** — used pizza boxes typically contaminated with grease/food residue, making them non-recyclable; grease soaking compromises biodegradability; also sometimes wax-coated |

**Verdict: CONFIRMED COMPOSITE PROBLEM — corroborates Phase 2b.**
The TACO subcategories `Meal carton` and `Pizza box` introduce coated/contaminated items
into the `cardboard` biodegradable class. The gcv2_/gc12_ catch-all `cardboard` categories
add Tetra Pak and other beverage cartons as noted in Phase 2b.

---

## Summary Table

| Class | Stage 1 Label | Clean? | Composite Subcategories | Stage 1 Impact |
|---|---|---|---|---|
| `battery` | Non-biodegradable | ✅ **Clean** | None | None |
| `glass` | Non-biodegradable | ✅ **Clean** | None | None |
| `metal` | Non-biodegradable | ⚠ One flag | `Aluminium blister pack` (taco_) | **Nil** — correctly non-biodegradable either way |
| `organic` | Biodegradable | ✅ **Clean** | None | None |
| `plastic` | Non-biodegradable | ⚠ One flag | `Other plastic` (taco_) — laminate pouches | **Nil** — correctly non-biodegradable either way |
| `textile` | Non-biodegradable | ⚠⚠⚠ **All subcategories flagged** | shoes, clothes, rope across all 3 sources | **HIGH** — natural-fibre items labelled non-biodegradable cause model confusion |
| `paper` | Biodegradable | ⚠⚠ **Multiple flags** | Magazine paper, Wrapping paper, Paper cup (taco_); catch-all gcv2_/gc12_ paper | **HIGH** — composite items labelled biodegradable drive 25.49% error rate |
| `cardboard` | Biodegradable | ⚠⚠ **Multiple flags** | Meal carton, Pizza box (taco_); Tetra Pak in gcv2_/gc12_ catch-all | **HIGH** — same mechanism, drives 20.69% error rate |

---

## Findings

### Classes with Stage 1 impact: `paper`, `cardboard`, `textile`
All three have structural composite ambiguity where specimens carry the wrong binary label.
The problem was confirmed visually for paper/cardboard (Phase 2b) and by error analysis
for textile (Phase 1). The mechanism is identical in all three: a source-dataset catch-all
category (or a specific TACO subcategory) files composite items under a biodegradable class.

### Classes without Stage 1 impact: `battery`, `glass`, `organic`
These are single-material classes with no composite subcategories. Their error rates in
the ResNet18 baseline confirm this: battery 7.69%, glass 3.00%, organic 4.49% — the
residual errors are genuine model difficulty (background clutter, unusual angles), not
taxonomy ambiguity.

### Classes with flags but no Stage 1 impact: `metal`, `plastic`
`Aluminium blister pack` and laminate `Other plastic` are composite items, but they feed
into non-biodegradable classes — so Stage 1 labels them correctly regardless. These matter
for fine-grained recycling routing (Stage 3) but do not affect binary accuracy.

---

## Recommendation

Three classes — `paper`, `cardboard`, `textile` — share the same root cause: source-dataset
catch-all labels that file composite materials under a single biodegradable target class.
The fix path is the same for all three and should be treated as a single coordinated
relabelling decision:

1. **`textile`**: Separate natural-fibre items (cotton/wool/linen clothes, jute rope) from
   synthetic-dominant items (polyester clothes, rubber-soled shoes, nylon rope). The
   non-biodegradable label is correct for the majority but incorrect for a significant
   minority.

2. **`paper`**: Flag or reclassify `Paper cup`, `Magazine paper`, and `Wrapping paper`
   TACO subcategories as non-biodegradable composite. Audit the gcv2_/gc12_ `paper` catch-all
   for the same specimens.

3. **`cardboard`**: Flag or reclassify `Meal carton`, `Pizza box`, and any Tetra Pak cartons
   in the gcv2_/gc12_ `cardboard` catch-all as non-biodegradable composite.

These three changes, if implemented together before retraining, are expected to resolve the
structural accuracy ceiling that persists even with a strong pretrained backbone.
