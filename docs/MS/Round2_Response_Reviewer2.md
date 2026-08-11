# Response to Reviewer #2 — Round 2

**Manuscript:** GENCIC / ASD circuits
**Comments source:** `docs/MS/ReviewComments_Round2.docx`
**Status tracker:** `docs/MS/Round2_Rebuttal_Tracker.md`
**Last updated:** 2026-08-11

> **Drafting status.** Major comment 2 (PD/HD validation) is drafted from completed analysis — every
> number is traceable to `results/PD_HD_validation/` and the four notebooks in
> `notebooks_disease_validation/`. All other responses are **placeholders awaiting analysis**.
> Reviewer text is verbatim from the comments file.

**Referee note.** Reviewer #2 states their expertise is "computational methods, genomics, genetics,
and autism phenotypes" and explicitly defers circuit-biology judgement to Reviewer #1. Responses to
this reviewer should therefore lead with statistics, controls and genetics rather than neuroanatomy.

---

## Reviewer #2 opening remarks

> This manuscript presents a new method for linking condition-associated genes to neural circuits in
> autism using mouse spatial transcriptomic and connectome data. This addresses an important gap in
> the field, where genetics and neural circuits are often studied separately and integrating these
> modalities remains a major challenge that limits our understanding of neurodevelopmental disorders.
> I should note that my expertise lies in computational methods, genomics, genetics, and autism
> phenotypes. Accordingly, I will largely leave evaluation of the neural circuit findings themselves
> to reviewers with greater expertise in that area.

**Response.** *[TODO — brief thanks; note that we have added a disease-validation section that speaks
directly to the computational/genetic concerns raised.]*

---

## Major comment 1 — Justification for mouse data

> The justification for using mouse data in this context is not sufficiently developed. Although the
> manuscript argues that psychiatric disease circuits are functionally and evolutionarily conserved
> across species, autism is a neurodevelopmental disorder with many human-specific features affecting
> phenotypes such as language and higher cognition that are difficult to model in mice. There should
> be a stronger explanation of why autism-relevant circuits, in particular, are expected to be
> sufficiently conserved for this framework. For example, how many of the ASD genes analyzed have
> established mouse models with ASD-relevant phenotypes? Is there evidence that the implicated mouse
> circuits overlap with circuits disrupted across existing ASD mouse models?

**Response.** ⬜ *TODO.* Needs: (a) count of analysed ASD genes with published mouse models showing
ASD-relevant phenotypes (candidate source: SFARI Gene animal-model annotations); (b) literature survey
of circuits disrupted across ASD mouse models vs our implicated structures; (c) an explicit concession
that human-specific phenotypes (language, higher cognition) are not modelled, with the argument
narrowed to what mouse data can support. Largely literature work; the Europe PMC curation pipeline
built for Major 2 is reusable.

---

## Major comment 2 — Validation on a disease with established genes and circuit 🟩

> The CCS/GENCIC framework is the primary methodological contribution of this work, so I would like to
> see stronger validation of its ability to recover biologically meaningful disease-associated
> circuits. The only validation presented in the main text is the dopamine circuit, with two
> additional neurotransmitter analyses in the supplement. These examples are reassuring sanity checks
> but do not fully validate the broader claim that disease-associated gene sets can recover
> disease-relevant neural circuits. Neurotransmitter systems are much more evolutionarily conserved
> than complex neurodevelopmental disorders. I would find it more convincing to validate the framework
> using a disease with both well-established causal genes and a well-characterized affected circuit,
> such as Parkinson's disease and/or Huntington's disease.

**Response.**

We thank the reviewer for this suggestion, which we agree goes to the heart of the paper's claim. We
have carried out the requested validation on Parkinson's disease and, as also suggested, on
Huntington's disease. The analysis was pre-registered, and we report both a positive and a negative
result. It is presented as new Supplementary Note section S-X, with supporting Supplementary Tables
S-PD1–S-PD5.

**Gene curation.** The primary set comprises 19 Mendelian Parkinson's/parkinsonism genes, each
traceable to a verified primary publication (Table S-PD1); eight are classified "Definitive" for
Parkinson disease by the ClinGen Gene–Disease Validity panel. Genes with disputed (`DNAJC13`,
`LRP10`), insufficient (`PTRHD1`) or unreplicated (`CHCHD2`) evidence, and genes whose syndromes are
pathologically distinct from nigrostriatal PD (`MAPT`, `DCTN1`, `POLG`, `TWNK`, `SPG11`, `DNAJC12`),
were excluded by a rule applied to the literature table and frozen before any analysis was run.

We also tested a common-variant tier: 41 genes assigned to Parkinson's GWAS loci by Open Targets
locus-to-gene scores rather than by proximity. This set does not recover the dopaminergic population
(AUROC 0.390, n.s.) or the affected structures, whereas the Mendelian set does. The framework
therefore appears to require gene sets with established causal relationships to the disorder; genes
nominated from common-variant association alone, even by principled locus-to-gene mapping, do not
carry the same spatial signal. We regard this as a useful delimitation of the method's input
requirements.

**Cell-type level.** Against the 43 dopaminergic clusters of the Allen Brain Cell Atlas
(`SNc-VTA-RAmb Foxa1 Dopa`), the PD gene set gives AUROC 0.986 (gene-set permutation p = 1×10⁻⁴).
Specificity is demonstrated against seven comparison sets, none significant on the same target: ASD
(0.384, p = 0.68), DDD (0.254, p = 0.89), and four non-brain traits (IBD, HDL-C, T2D, HbA1c;
p = 0.30–0.84). The DDD gene set instead recovers striatal medium spiny neurons (p = 0.031) while the
PD set does not (p = 0.60) — a double dissociation in which each disease gene set recovers its own
cell population and not the other's.

**Structure level.** Fifteen structures reach FDR q < 0.10, led by ventral tegmental area, dorsal raphe
and substantia nigra pars compacta (all q ≈ 0.004). To test whether the ranking recovers structures
genuinely affected in PD, two independent raters classified 50 brainstem and diencephalic structures
for documented human PD pathology, blind to our results, with the comparison set drawn exclusively from
the same anatomical territory so that "it is a brainstem nucleus" carries no information. Structures
judged PD-affected rank substantially higher in our ranking than unaffected structures (median rank 12
vs 54; rank-sum p = 0.0025 for consensus calls, p = 0.0006 under the lenient criterion; inter-rater
κ = 0.59).

**Circuit level.** Applying the same Pareto criterion used for the ASD circuit in the main text — the
operating point at which a ~20% reduction in mean bias buys the largest gain in circuit connectivity —
the recovered circuit gains 77–137% CCS across circuit sizes 11–20 and retains substantia nigra pars
compacta, ventral tegmental area and three raphe nuclei at every size. Nine structures are common to
all recovered circuits (Table S-PD4).

**A negative result we report in full.** Our pre-registered composite test — whether the 13 structures
of the canonical PD motor circuit collectively rank above the remaining 200 — is not significant
(AUROC 0.582, p = 0.11). The reason is informative rather than technical. That list conflated the
structures in which neurons degenerate with the structures whose function is disrupted downstream. The
caudate–putamen is severely dopamine-depleted in PD but its own neurons are largely preserved, and our
method ranks it 190th of 213. GENCIC identifies where vulnerable cells reside, not where the circuit
malfunctions; on that reading the low rank of the striatum is the correct answer rather than a failure.
We report both the pre-registered composite and the blinded pathology analysis, and state which was
specified in advance.

**Huntington's disease.** We also tested HD, and it does not work. `HTT` is depleted, not enriched, in
the medium spiny neurons that degenerate (all four MSN subclasses rank 278–292 of 340 subclasses), and
an eight-gene panel of Mendelian striatal-degeneration syndromes likewise fails (AUROC 0.371, n.s.). We
regard this as a scope boundary rather than a technical failure, consistent with the long-standing
observation that HTT is ubiquitously expressed and that its expression does not explain the selective
vulnerability of striatal neurons. Current understanding attributes MSN vulnerability to somatic
CAG-repeat instability, to striatum-enriched interacting partners such as RASD2/Rhes, and to
non-cell-autonomous corticostriatal and glial contributions — none visible in a baseline expression
atlas. GENCIC identifies cell populations whose expression profile is enriched for risk genes; it
therefore captures PD-type biology and is not expected to capture HD-type biology. We believe stating
this boundary explicitly strengthens rather than weakens the framework's claims.

*Supporting analysis: notebooks `01`–`04` in `notebooks_disease_validation/`; tables and figures in
`results/PD_HD_validation/`; pre-registration commits `b79558d` and `a3ae114`.*

---

## Major comment 3 — What does a single "ASD circuit" mean?

> I found the discussion of a single "ASD circuit" conceptually confusing. As I understand it, this
> circuit is generated by aggregating mutation signals across dozens of ASD genes identified through
> rare variant analyses. However, the majority of individuals with autism do not carry highly penetrant
> variants in any of these genes, and those who do typically harbor only a single damaging variant. Is
> the claim that many ASD genes converge on a common circuit, or that this represents an average
> population-level vulnerability landscape? More discussion of how the substantial genetic and
> phenotypic heterogeneity of autism fits into this framework would improve the interpretation of these
> results.

**Response.** ⬜ *TODO.* The reviewer is asking us to commit to one of two claims. Decide explicitly
whether we are asserting convergence or an average vulnerability landscape, then support it. The
existing gene-clustering analysis may be able to distinguish these. Note the PD result is relevant
context: there, 19 genes acting through several distinct molecular mechanisms nonetheless converge on
one cell population — an argument that convergence is possible without uniform mechanism.

---

## Major comment 4a — Why expression specificity rather than overall expression?

> First, it is not clear why expression specificity, rather than overall expression, is the feature
> emphasized throughout the analysis. Is there biological evidence supporting the idea that
> region-specific expression should be particularly informative for identifying autism-associated
> circuits?

**Response.** ⬜ *TODO.* Needs a biological justification for the Z2 specificity measure, and ideally a
supporting analysis repeating the main result with overall expression to show it is less informative.

---

## Major comment 4b — Is the ASD>sibling result built into the metric?

> Second, Figure 2 and the surrounding text conclude that ASD mutation biases are stronger in autistic
> individuals than in unaffected siblings. However, this appears at least partly built into the
> construction of the metric. My understanding is that the brain structure scores are already weighted
> by enrichment of disorder-associated mutations, making stronger ASD enrichment in the output somewhat
> expected. The authors should clarify which analyses are independent of this weighting and whether
> alternative null models produce similar conclusions.

**Response.** ⬜ *TODO — but the machinery exists.* Two parts. (i) State plainly which analyses are
independent of the mutation weighting. (ii) Report ASD under alternative nulls. For the PD validation
we implemented four: uniform random, expression-decile-matched, sibling-uniform, and
sibling-mutability-weighted (the last matching the published ASD procedure). Conclusions were unchanged
across all four. The same four can be run for ASD with existing code —
`scripts/script_generate_geneweights.py` (`expmatched` branch) and `Snakefile.bias`.

⚠️ *Rerun hazard:* editing `rule generate_geneweights` marks existing targets stale under Snakemake's
default rerun-triggers. Use `--rerun-triggers mtime` and explicit targets so published nulls are not
regenerated.

---

## Major comment 5 — Is the circuit too large to be specific?

> Figure 5b, together with the overall description of the identified circuitry, gives the impression
> that a substantial fraction of the brain is implicated. This raises questions about the specificity
> of the resulting circuit. Is it expected that such a large proportion of the brain would show
> enrichment? Discussion of the dynamic range of the mutation bias values and the specificity of the
> identified structures would be helpful. Given the complexity and heterogeneity of autism, many brain
> regions could plausibly be linked to the disorder post hoc, so strong statistical controls are
> particularly important. Is a circuit of this size typical for complex neurological disorders? Is
> there a smaller "core" circuit that can be identified with higher confidence?

**Response.** ⬜ *TODO.* Needs: dynamic range of bias values; whether a 46-structure circuit is typical;
and identification of a smaller high-confidence ASD core. The PD analysis supplies a direct comparator
for the reviewer's question about typical size — a 13-structure circuit with a 9-structure core stable
across sizes 11–20 — and the consensus-core method transfers directly to ASD.

---

## Minor comment 1 — "Independent" validation gene sets overlap SPARK

> The gene sets used as independent validation (Satterstrom et al. and Fu et al.) overlap substantially
> with the SPARK gene set. The language describing these analyses as independent should therefore be
> tempered.

**Response.** ⬜ *TODO.* Compute and report the exact overlap; soften "independent validation" wording
throughout. The reviewer is right and this should simply be conceded.

---

## Minor comment 2 — Interpreting CCS

> I found the Circuit Connectivity Score (CCS) somewhat difficult to interpret. What does the maximum
> CCS represent biologically? How should readers interpret absolute CCS values? Why does the peak CCS
> determine the preferred circuit size?

**Response.** ⬜ *TODO.* Explain the biological meaning of maximum CCS, how to read absolute values, and
why peak CCS sets circuit size. Note from the PD work: three independent criteria (FDR count under two
nulls, and the CCS profile peak) converged on the same circuit size (~13), which is supporting evidence
that the peak-CCS criterion is not arbitrary.

---

## Minor comment 3 — Are the Pareto objectives genuinely competing?

> Pareto fronts are typically used to optimize competing objectives. In this case, it is not obvious
> that anatomical connectivity and mutation bias are truly competing quantities. Was Pareto
> optimization simply chosen as a convenient multi-objective optimizer, or is there a biological
> rationale for treating these objectives as competing?

**Response.** ⬜ *TODO — but we now have direct evidence.* They are genuinely competing, and the PD
analysis demonstrates it concretely. Along the PD Pareto front, at the permissive end the optimiser
sacrifices the highest-bias structures (substantia nigra, VTA) in favour of the densely
interconnected reticular formation, more than doubling CCS while halving mean bias; at the
ASD-matched operating point (−20% bias) it retains the dopaminergic core. The objectives therefore
trade against one another, and the choice of operating point is consequential — which is precisely
what a Pareto formulation is for. Figure `04b_pareto_fronts_operating_points.png`.

---

## Minor comment 4 — Same metric used for optimization and evaluation

> The same mutation bias metric is used throughout multiple stages of the framework, including ranking
> brain structures, selecting circuit size, optimizing the circuit, and evaluating the resulting
> circuit. It would strengthen confidence in the results if some aspects of optimization and evaluation
> relied on more independent data or metrics, for example through held-out validation.

**Response.** ⬜ *TODO — partly answered by Major 2.* The PD/HD validation evaluates the framework
against a criterion entirely independent of the bias metric: published human neuropathology, assessed
by two raters blind to our rankings, with anatomically matched controls. Consider additionally a
held-out validation for ASD (e.g. train circuit selection on one cohort, evaluate on another).

---

## Drafting checklist

- [x] Major 2 — drafted from completed analysis
- [ ] Opening remarks
- [ ] Major 1, 3, 4a, 4b, 5
- [ ] Minor 1, 2, 3, 4
- [ ] Consistency pass against Reviewer #1 responses (especially R#1-1 terminology, R#1-7/9 matched nulls)
- [ ] Final numbers re-checked against `results/PD_HD_validation/` before submission
- [ ] Supplementary table S-numbers assigned
