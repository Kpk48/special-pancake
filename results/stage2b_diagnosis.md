# Phase 2b — Paper/Cardboard Hard Error Inspection Report

## 1. Cross-Tab: Source × Fine Class

| Fine Class | Source Tag | Total Images | Error Count | Error Rate |
| --- | --- | --- | --- | --- |
| `paper` | `gc12_` | 138 | 29 | 21.01% |
| `paper` | `gcv2_` | 108 | 28 | 25.93% |
| `paper` | `taco_` | 9 | 8 | **88.89%** |
| `cardboard` | `gc12_` | 125 | 17 | 13.60% |
| `cardboard` | `gcv2_` | 107 | 31 | **28.97%** |
| `cardboard` | `taco_` | 0 | 0 | — |

### Are paper/cardboard errors concentrated in a specific source tag, or spread evenly?

The errors are **not concentrated in a single source tag** — they are spread across gc12_ and gcv2_ at comparable absolute counts (paper: 29 vs 28, cardboard: 17 vs 31). However there is a strong **severity gradient** by source:

- `taco_` paper has by far the highest error rate (88.89%) but tiny sample count (9 images), so its absolute contribution is small (8 errors).
- `gcv2_` is the dominant absolute error contributor for cardboard (31 of 48 errors) and nearly tied for paper.
- `gc12_` has lower rates but large volume means it still contributes many errors.

**Conclusion on concentration:** This is neither pure domain-shift (which would manifest only in taco_) nor evenly distributed noise. It reflects a **distributed taxonomy ambiguity** where paper and cardboard in both studio datasets (gc12_ and gcv2_) contain visually non-biodegradable specimens. The taco_ spike confirms real-world examples make this worse, but the core problem exists even in controlled studio images.

---

## 2. Visual Inspection of the 20 Highest-Confidence Hard Errors

The contact sheet at `results/stage2b_paper_cardboard_contact_sheet.png` was inspected. Each image is classified as either:
- **Plain** — genuinely plain paper or cardboard (model failure or label correct but visually tricky)
- **Composite/Contaminated** — visible coating, lamination, mixed materials, printing ink, food staining, plastic components, metallic lining, or wax that makes "biodegradable" a questionable label

| # | Filename | Class | Source | Conf | Assessment |
|---|---|---|---|---|---|
| 1 | `gc12_paper_a01cdc.jpg` | paper | gc12_ | 92.7% | **Composite** — printed magazine/book pages with glossy coating and metallic staples visible |
| 2 | `gcv2_paper_3eab2b.jpg` | paper | gcv2_ | 91.1% | **Composite** — crumpled newspaper on dirty surface with adhesive tape and food staining |
| 3 | `gcv2_paper_c59fc0.jpg` | paper | gcv2_ | 90.3% | **Plain** — crumpled blue tissue paper; model failure, label correct |
| 4 | `gcv2_cardboard_5eca2d.jpg` | cardboard | gcv2_ | 90.1% | **Composite** — purple printed cardboard packaging with laminated barcode sticker and visible plastic film window |
| 5 | `gc12_paper_6a6060.jpg` | paper | gc12_ | 89.0% | **Composite** — printed newspaper/magazine page visible inside what appears to be a used plastic-lined bag |
| 6 | `gcv2_paper_0947cc.jpg` | paper | gcv2_ | 85.6% | **Composite** — paper cups with wax/plastic lining (visibly food-contaminated, inside a bin with plastic cups) |
| 7 | `taco_Paper_bag_0c05a0.jpg` | paper | taco_ | 84.7% | **Composite** — large paper bag on ground with prominent red/blue printed branding, foil-style printing, heavily soiled |
| 8 | `gcv2_paper_5c5de2.jpg` | paper | gcv2_ | 84.1% | **Composite** — shredded mixed paper waste pile containing visible plastic film strips and metallic packaging fragments |
| 9 | `taco_Paper_bag_db87e6.jpg` | paper | taco_ | 84.1% | **Plain** — crumpled brown kraft paper bag on rough terrain; plain material, model failure |
| 10 | `gc12_cardboard_485802.jpg` | cardboard | gc12_ | 83.1% | **Composite** — angular cardboard piece showing a metallic/reflective inner liner (Tetra Pak or foil-lined packaging) |
| 11 | `gc12_cardboard_5b3ee8.jpg` | cardboard | gc12_ | 81.2% | **Plain** — clean flat white cardboard sheet with simple recycling icons; model failure, label correct |
| 12 | `gcv2_paper_4aae78.jpg` | paper | gcv2_ | 80.7% | **Composite** — cardboard tube with visible adhesive and plastic sealing tape |
| 13 | `gc12_paper_8c83f7.jpg` | paper | gc12_ | 79.9% | **Composite** — newspaper roll tightly bound with plastic bag/film wrap |
| 14 | `gcv2_paper_2f8726.jpg` | paper | gcv2_ | 79.2% | **Composite** — large shredded paper pile with clear plastic and metallic packaging fragments mixed in |
| 15 | `gc12_paper_1dcce3.jpg` | paper | gc12_ | 79.1% | **Composite** — snack packet (metallic foil-lined food packaging) labelled paper but clearly a composite laminate |
| 16 | `gc12_paper_2bc727.jpg` | paper | gc12_ | 78.8% | **Composite** — product box with prominent plastic windows and multi-layer printed laminate |
| 17 | `taco_Paper_bag_4acb79.jpg` | paper | taco_ | 77.3% | **Plain** — plain brown kraft paper bag on outdoor surface; model failure, label correct |
| 18 | `gcv2_cardboard_553f21.jpg` | cardboard | gcv2_ | 77.1% | **Composite** — Alpro oat milk Tetra Pak carton: cardboard exterior but inner layers are polyethylene and aluminium foil |
| 19 | `gcv2_paper_e5e2ba.jpg` | paper | gcv2_ | 76.9% | **Plain** — crumpled white A4 paper sheet; plain material, model failure |
| 20 | `gc12_paper_3c2649.jpg` | paper | gc12_ | 76.5% | **Composite** — Citi Bank branded cardstock mailer envelope with visible plastic window and synthetic lining |

**Summary tally:** 15 of 20 images show visible composite/contaminated material, 5 are plainly biodegradable paper or cardboard where the model is failing on clear specimens.

---

## 3. Recommendation

### What fraction show visible mixed-material contamination?

**15 out of 20 (75%)** of the highest-confidence hard errors show one or more of: wax/plastic coating, laminate layers, plastic film mixed into the pile, metallic foil lining, food contamination, adhesive tape, plastic windows, or glossy print stock. These specimens are only labelled `paper` or `cardboard` because their outer shell is paper-based, but they are composite materials not straightforwardly biodegradable.

Notable recurring patterns:
- **Tetra Pak / Juice cartons** (e.g. `gcv2_cardboard_553f21.jpg`) — cardboard shell, polyethylene + aluminium inner layers
- **Food-service paper cups** (e.g. `gcv2_paper_0947cc.jpg`) — wax or plastic-lined interiors
- **Printed packaging with plastic windows** (e.g. `gc12_paper_3c2649.jpg`, `gc12_paper_2bc727.jpg`)
- **Shredded mixed waste** containing visible plastic/foil mixed in (e.g. `gcv2_paper_2f8726.jpg`, `gcv2_paper_5c5de2.jpg`)

### Is this (a) a taxonomy problem, (b) model confusion needing more data, or (c) noise?

This is **primarily (a) a taxonomy problem**, with a secondary component of **(b) model confusion on real-world specimens**.

The evidence:
1. **75% of hard errors are not mislabels in the conventional sense** — they are specimens that the dataset sources consistently placed in `paper` or `cardboard` folders because their outer material is paper-based, but the model's features (recognising plastic film, foil reflectance, beverage carton shapes, glossy print) correctly identify them as visually non-biodegradable composites. The model is arguably *more correct* than the label on these samples.
2. **Only 25%** (5 images) show plainly biodegradable material where the model is clearly wrong — these represent genuine model failure, likely due to background clutter (real-world TACO photos) or unusual orientation.
3. **The pattern parallels the textile finding from Phase 1** — just as `textile` contained both natural-fibre (biodegradable) and synthetic (non-biodegradable) garments under one label, `paper` and `cardboard` contain both plain paper (genuinely biodegradable) and composite packaging (not biodegradable) under one label.
4. **Domain-shift explains the taco_ spike** (88.89% error rate) but not the majority of gc12_ and gcv2_ errors — those are taxonomic, not photographic noise.

**Recommended action:** Introduce a handling rule analogous to the textile issue — consider splitting the `paper`/`cardboard` taxonomy by whether the item is **plain paper/cardboard** vs. **composite packaging** (Tetra Pak, coated cups, plastic-windowed boxes, laminated cartons). This would either be a new fine-grained class or a reclassification of known composite specimens to a non-biodegradable group. Simply adding more training data of the same labelling policy will not resolve this; the ambiguity is structural.
