# Rule-Grounded Recommendation Categories

This guide explains the eight deterministic questions used by the active-learning dashboard. A category is not an LLM opinion. Code first decides whether a typical case exists, builds the candidate pool, and fixes the ordered recommendation points. DeepSeek V4 Pro may explain that fixed plan but cannot change it.

## How To Read A Category

Each available category should answer four practical questions:

- Which records should I label now?
- What visible pattern makes these records worth checking?
- What should I compare when deciding the label?
- What would each possible answer teach the next round?

If no defensible case exists, the category is shown as unavailable. The system does not substitute unrelated records.

For every recommended record, the page shows the same five-part structure:

- Why this category: the labeling question being checked.
- Why this record was recommended: every fixed evidence question, its direct
  answer, and what this record's human label would clarify.
- Compare with: clickable human-confirmed examples when available, otherwise clearly marked system examples.
- How to label it: the records and original fields the user should compare.
- What your answer would tell us: no more than two plain if-then outcomes.

Each evidence check has a visible status: yes, partly, no, or not enough
evidence. The main explanation avoids scores, distances, percentages, and
rule thresholds. Those calculations remain available in folded Technical
details.

## 1. Label Priority

### What it examines

Label Priority decides which unresolved labeling question is most useful to address next.

### How it is determined

It compares the available category plans using unresolved human/model disagreement, the number of records that may be affected, candidate availability, and how often similar questions were already shown. Category order and point ID provide stable tie-breaking.

### Evidence checklist shown for every point

- Why this question comes first: identifies the unresolved question and why it matters now.
- Why this point is a clear example: explains whether it is especially close to the uncertain area, has a clear conflict, or has useful comparisons.
- Whether one label answers several questions: states which other labeling questions this record can help check.
- Whether this is a new check or a recheck: explains whether the record is new, remained unresolved, moved groups, or changed unusual status.
- The complete checklist of the delegated category: Label Priority can never replace a real explanation with "high priority."

### What the user labels

The records from the highest-priority available plan, usually a small mixed batch that can resolve a current boundary, anomaly, or rule doubt.

### How to decide

Use the selected plan's ordinary-language question. Confirm an existing real-world type, create a new type when the existing vocabulary is insufficient, mark unusual/normal where appropriate, or defer when evidence is genuinely unclear.

## 2. Boundary Review

### What it examines

Boundary Review checks records that sit close to a rule-defined dividing line between current groups.

### How it is determined

The engine finds neighboring rule regions and ranks eligible records whose source-feature values are closest to the relevant threshold, while preserving batch diversity.

### Evidence checklist shown for every point

- Is it close to another group?
- Is it at the edge of its own group?
- Is it close to a rule's dividing line?
- Are similar nearby records divided between groups?
- Does the 2D plot tell the same story as the complete feature space?

The last check prevents the user from trusting a visually close point when the
original multidimensional record tells a different story.

### What the user labels

Records on both meaningful sides of the same dividing line. The batch is designed to test the line, not merely collect several almost identical records.

### How to decide

Compare records across the line using their original fields and domain meaning. If similar records deserve the same human label, the current division may be misplaced. If their labels differ consistently, the boundary gains support.

## 3. Overlap Merge Signal

### What it examines

This category asks whether two rule regions cover many of the same records and may represent one shared concept or an unclear boundary.

### How it is determined

The engine compares rule membership and only activates when overlap is substantial enough to form a typical case. It recommends records from the shared region and, when useful, contrasting records outside it.

### Evidence checklist shown for every point

- Does it fit important parts of both group descriptions?
- Do its most similar records come from both groups?
- Does it resemble typical examples from both groups, or clearly favor one?
- What do existing human labels say in the shared area?
- Are the groups still different away from this local overlap?

No human labels is shown as insufficient evidence, never as evidence for a
merge. The page may recommend checking whether two descriptions represent the
same type, but it never performs or directly recommends a merge.

### What the user labels

Representative records shared by the two rule regions.

### How to decide

If the shared records consistently receive one human label, a merge or shared-boundary review becomes plausible. If users can reliably distinguish two real-world types inside the overlap, merging would erase useful structure. No merge is executed automatically.

## 4. Split Or New Cluster Signal

### What it examines

This category checks whether a current group contains separated, weakly covered, or internally inconsistent regions that may need different human concepts.

### How it is determined

It looks for disjoint rule regions, weak rule coverage, coherent exceptions, and separated candidate groups. It activates only when there is enough evidence for a meaningful comparison.

### Evidence checklist shown for every point

- Is it separated from the typical members of its current group?
- Is it isolated, or part of a small coherent pocket?
- Does it resemble an existing group?
- Do nearby human labels support one shared type?
- Has the separation remained visible across rounds?

A first-round separation is explicitly described as early evidence. The user
is asked whether the record fits an existing type or has a genuinely different
meaning; the page never orders the system to create a cluster.

### What the user labels

Records from the suspected subregions, chosen so the batch compares the possible divisions.

### How to decide

Repeatedly different human labels support a split or a new semantic class. The same label across subregions suggests the current group may be broad but still meaningful. The dashboard proposes review; it does not create a cluster by itself.

## 5. Anomaly Label Review

### What it examines

Anomaly Label Review checks whether records currently flagged as unusual are true domain anomalies or valid rare members.

### How it is determined

It combines current outlier status, anomaly rules, proximity to rule conditions, and available normal contrasts. Previously labeled records return only when their status changed or conflicts with current evidence.

### Evidence checklist shown for every point

- Is it unusual compared with its own group?
- Is it isolated, or part of a repeatable rare pattern?
- Could missing, extreme, or malformed fields explain the difference?
- Has its unusual status changed across rounds?
- How does it compare with human-confirmed normal or unusual examples?

When confirmed examples do not exist, the page says so and may provide a
clearly marked system example only as a comparison aid.

### What the user labels

A small set of flagged records and, when informative, nearby normal-looking comparisons.

### How to decide

Mark true outlier when the record is genuinely invalid, exceptional, or outside the domain pattern. Mark normal when it is rare but legitimate. Use uncertain when the available fields cannot support the distinction.

## 6. Exception Relabel Review

### What it examines

This category reviews records that do not fit the rule that usually describes their current group.

### How it is determined

The engine identifies RuleSet exception points and ranks those with the strongest conflict, greatest potential impact, and least recent review.

### Evidence checklist shown for every point

- Which part of its current group description does it not fit?
- Is the disagreement just outside the description or clearly different?
- Do similar nearby records support its current group, another group, or a mixed result?
- Does it resemble another group's typical examples?
- Is this one exceptional record, or does the same rule fail for many similar records?

If a rule fails repeatedly in the same area, the explanation questions the
simple rule before questioning every record's label.

### What the user labels

Records whose current assignment and rule-based description disagree.

### How to decide

If the human label agrees with the current group, the surrogate rule is too simple and should not be trusted for that case. If the human label disagrees with the group, the assignment or boundary deserves review. The tree remains explanation-only in either outcome.

## 7. Feature Label Strategy

### What it examines

Feature Label Strategy asks which original fields provide the clearest checklist for labeling the current recommended records.

### How it is determined

It uses features that repeatedly appear in relevant rules, then selects records that make those feature conditions informative. Internal scaled or one-hot columns are decoded into source fields.

### Evidence checklist shown for every point

- Why is this original field being examined?
- Is the record clearly on one side of the field's dividing line or near it?
- Does that field agree with the story told by the whole record?
- Do existing human labels follow the same pattern?
- Could the field be a shortcut caused by missingness, duplication, or a pattern unique to the current sample?

The field is always a checking clue, never an automatic label. The user labels
the complete record.

### What the user labels

Records that test whether the rule's source-feature story matches the user's real-world concept.

### How to decide

Treat the listed source fields as comparison prompts, not unquestionable truth. Agreement supports the checklist. Disagreement means the current rules omit important context or the human concept uses different information.

## 8. Rule Confidence Audit

### What it examines

Rule Confidence Audit checks whether a rule is reliable enough to guide future labeling explanations.

### How it is determined

It considers coverage, consistency, support, warnings, and exceptions, but presents these as qualitative strengths or limitations in the main UI. Records are chosen where an audit can confirm or challenge the rule.

### Evidence checklist shown for every point

- Does the rule describe a small part, a substantial part, or most of the group?
- Does it usually match the current analysis, sometimes fail, or often miss cases?
- Does it agree with existing human labels?
- What kinds of records break it?
- Have its fields, direction, and dividing line remained stable across rounds?

The main card uses these qualitative phrases rather than raw coverage or
purity values.

### What the user labels

Representative matched records and important exceptions for the target rule.

### How to decide

Consistent labels across both groups strengthen trust in the rule. Repeated disagreement limits the rule's explanatory value and may point to a missing feature, an overly broad condition, or unstable analysis.

## What DeepSeek Adds

DeepSeek receives only the fixed recommendation plan, relevant rule summaries,
recommended record profiles, fixed evidence checks, allowed labels, prior
label context, and round changes. It translates supplied observations and
their significance into plain instructions and conditional outcomes.

It cannot choose points, change their order, omit or reorder checks, change a
check's status, invent reasons, replace comparison records, alter groups or
unusual status, or commit labels. An invalid bullet is replaced with its
deterministic wording; a provider failure uses the complete deterministic
guidance, so the active-learning round remains usable.
