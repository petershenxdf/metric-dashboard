# Rule Interpretation Categories Guide

生成时间：2026-06-11

这份文档解释 Rule Panel 中八个 interpretation category 的含义。它们不是普通聊天分类，也不是新的 clustering / outlier detection 算法。它们的共同目标是：把 decision-tree surrogate rules 转换成用户下一步应该如何 label、如何检查边界、如何判断是否需要 merge / split / new cluster / anomaly review 的具体建议。

## 1. 背景：这些 category 在系统里做什么

当前系统的责任边界是：

- SSDBCODI 负责当前 cluster assignment 和 outlier flags。
- Decision tree 只负责把 SSDBCODI 的输出写成可读 rule，不负责重新 clustering，也不负责重新判断 outlier。
- Rule interpretation 负责把这些 rule 变成 label/refinement guidance。
- DeepSeek 或 mock interpreter 不能直接修改 cluster、outlier、selection 或 labeling state。

换句话说，interpretation category 回答的是：

- 哪些点应该先被用户 label？
- 哪两个 rule 的关系需要人工确认？
- 当前 evidence 支持 merge、split、new cluster，还是只支持继续观察？
- 哪些 anomaly / exception points 应该优先确认？
- 目前 rule 的可信度够不够支持用户做下一步 refinement？

## 2. 输出结构

每次 interpretation 至少包含这些字段：

- `categories`: 一个或多个 category id。
- `category_explanation`: 一小段说明这个 category 主要考察什么。
- `recommendation`: 一句话说明用户下一步最应该做什么。
- `summary`: 更详细的解释，说明为什么这样做。
- `label_targets`: 明确建议用户 label 哪些 point，以及要回答什么 label question。
- `suspicion_reasons`: 解释为什么这些 points 有疑点或值得优先检查。必须同时包含 rule-level reason 和 point-level reason。
- `point_label_guidance`: 说明这些 points 应该怎么 label，包括可选 label frame、检查 checklist、不同结果的 decision impact。这是 DeepSeek LLM 的主要价值输出。
- `decision_rationale`: 解释为什么这个 action 在策略上有价值。它不只是重复数字，而是说明这些 label 如何测试 merge、split、new cluster、anomaly 或 boundary hypothesis。
- `label_outcomes`: 用户给出不同 label 结果时，系统应该如何解读。例如 labels agree 可能支持 merge review，labels disagree 可能支持保留 boundary。
- `quantitative_findings`: 数字证据，例如 support、coverage、purity、Jaccard overlap、exception count。
- `suggested_label_actions`: 可以执行的 label action，例如 inspect points、audit rule、confirm anomaly。每个 action 必须包含 `reason`、`hypothesis`、`why_this_action`、`expected_outcomes` 和 `risk_note`。
- `evidence`: rule id、feature threshold、point id、target id 等可追溯证据。
- `warnings`: 解释限制，例如没有 sample-level overlap、没有 exception points。

## 2.1 LLM 的真正作用

系统先用代码计算 quantitative metrics，这是为了保证数字可靠、可审计、不会让 LLM 自己编造 overlap 或 support。LLM 的价值不在于重新计算这些数字，而在于把数字转化成下一步分析策略：

- 把 rule overlap、boundary gap、exception points 解释成可测试的 hypothesis。
- 说明为什么某些点比随机点更值得 label。
- 结合 point-level raw feature values、threshold margins、当前 cluster/outlier 状态，解释这些 point 到底应该如何 label。
- 给出不同 label outcome 对应的决策含义。
- 指出不确定性和风险，例如 sample-level overlap 为 0 时不能直接建议 merge。
- 把 raw feature thresholds 转成用户能执行的 labeling checklist。

因此，一个好的 interpretation 不应该只是 “Jaccard = 0.20”，而应该是：“这两个 rule 有 0.20 Jaccard overlap；建议先 label wine_014 和 wine_021，因为它们同时测试 rule overlap 和边界语义。如果这两个点被标成同一类，merge/shared-boundary review 变得合理；如果 labels mixed，则更可能是 boundary ambiguity 或 new cluster signal。”

特别注意：当系统建议 “inspect raw feature thresholds” 时，并不是说 raw feature 本身有问题。真正的疑点是：decision tree 的 threshold 只证明它能复现 SSDBCODI 当前输出，不证明这个 threshold 对用户的语义 label 也正确。因此用户需要检查这些候选点的 raw feature values 和 threshold margins 是否真的对应人类语义。

## 3. 常用指标解释

### Support

`support_count` 表示某条 rule 匹配了多少个点。support 越大，这条 rule 涉及的区域越大。高 support 的 rule 适合作为初始审查对象；低 support 的 rule 常常适合做 anomaly 或 exception inspection。

### Coverage

`coverage` 表示 rule 覆盖目标类别的比例。例如一条 cluster rule 的 coverage 为 0.70，表示该 cluster 里大约 70% 的目标点被这条 rule 覆盖。coverage 高说明 rule 对该目标有代表性；coverage 低说明该 cluster 可能需要多条 rule 才能解释，可能暗示 split / subregion review。

### Purity

`purity` 表示 rule 匹配的点里有多少比例属于 rule 的目标 target。purity 高说明 rule 匹配区域较干净；purity 低说明该 rule 可能混入其他 cluster 或 normal / anomaly 点，应该优先人工 label。

### Rule Confidence Score

`rule_confidence_score = purity * coverage`。它不是模型概率，而是一个排序信号。高分 rule 适合作为稳定区域的代表；低分 rule 应该被审查，不能直接作为 merge / split 依据。

### Exception Count / Exception Rate

`exception_count` 是被 rule 匹配但不属于 rule 目标 target 的点数。exception point 是最值得人工 label 的点之一，因为它们直接说明当前边界或 label 存在不一致。

### Pair Intersection Count

`intersection_count` 表示两条 rule 匹配点集合的交集大小。它是 overlap merge review 的核心证据。不同 target 的 rule 如果共享很多点，说明这些点可能处于边界、混合区域，或者当前 cluster 定义不稳定。

### Jaccard Overlap

`jaccard_overlap = intersection / union`。它衡量两条 rule 匹配集合的相似程度。Jaccard 越高，两条 rule 描述的样本区域越接近。注意：如果 sample-level overlap 为 0，就不能仅凭 rule 文本建议 merge。

### Overlap Share A / B

`overlap_share_a` 和 `overlap_share_b` 分别表示交集占 rule A / rule B 的比例。某条小 rule 如果大部分被大 rule 覆盖，可能说明小 rule 是大 rule 的子区域，也可能说明它是异常或边界切片。

### Boundary Gap

`boundary_gap` 描述两条 rule 在共享 feature threshold 上是否相邻、重叠、或分离。如果 gap 很小或为 0，说明这两条 rule 在 raw feature 空间里相邻，适合 boundary review。

## 4. Category 1: Label Priority

### 它回答什么

Label Priority 负责回答：“用户现在最应该先 label 哪些点？”它是 overview 或默认 interpretation 最常用的 category。

### 什么时候出现

当系统需要给出一个全局下一步，而不是只解释某一种特殊情况时，就应该使用 Label Priority。它通常综合所有 rule、pair metrics、candidate groups、exception points，挑出最值得人工 label 的点或区域。

### 它会看哪些证据

- highest-priority rule pair。
- label candidate groups。
- support、coverage、purity。
- exception count。
- overlap 或 boundary relation。
- representative matched points。

### 它应该输出什么

Label Priority 的输出应该明确说：

- 先 label 哪几个 point id。
- 这些点来自哪些 rule。
- 为什么这些点比随机点更有价值。
- label 之后可能影响 merge、split、new cluster，还是 anomaly review。

### 用户应该怎么用

用户可以把它当成主动学习策略的第一步：先 label 系统认为信息量最高的点，而不是从 scatterplot 上随便点。

### 典型例子

如果 strongest rule 覆盖 43 个点，purity = 1.00，coverage = 1.00，同时没有 exception，那么 Label Priority 可能建议先 label 这条 rule 的 representative points，用来确认这个区域确实是稳定 cluster。  
如果另一个 rule pair 有 cross-cluster overlap，它会优先建议 label overlap points，而不是 label 稳定区域。

### 注意事项

Label Priority 不等于自动修改模型。它只决定 “先看哪里”。是否 merge / split / relabel，必须等用户 label 之后再做。

## 5. Category 2: Boundary Review

### 它回答什么

Boundary Review 负责回答：“两个 rule 所描述的区域边界是否值得人工检查？”

### 什么时候出现

当两条 cluster rule 在 raw feature threshold 上相邻，或者两条 rule 的 boundary gap 很小，或者不同 target 的 rule 在特征空间里接近时，应使用 Boundary Review。

### 它会看哪些证据

- shared feature names。
- threshold interval。
- boundary_gap。
- pair relation，例如 `adjacent_cluster_boundary`。
- 两侧 rule 的 support、coverage、purity。
- 边界附近 candidate point ids。

### 它应该输出什么

Boundary Review 应该说明：

- 哪两条 rule 构成边界。
- 边界由哪些 raw feature 和 threshold 构成。
- 应该 label 边界两侧哪些点。
- 如果两侧用户 label 一致，可能说明 boundary 过度切分。
- 如果两侧用户 label 不一致，说明当前 boundary 可能是合理的。

### 用户应该怎么用

用户应该成对 label：不要只 label 一侧。边界问题需要比较边界两边的点是否在语义上属于同一组。

### 典型例子

如果 rule A 是 `proline <= 484`，rule B 是 `proline > 484 and proline <= 645`，它们在 `proline = 484` 附近相邻。Boundary Review 会建议用户在两个 rule 各抽几个点 label，看这个 threshold 是否真的对应一个 cluster boundary。

### 注意事项

Boundary Review 不直接推荐 merge。它只说 “这个边界值得检查”。merge 需要用户 label 证明两侧语义一致。

## 6. Category 3: Overlap Merge Signal

### 它回答什么

Overlap Merge Signal 负责回答：“两条 rule 是否因为共享点而可能表示同一个 cluster 或同一个语义区域？”

### 什么时候出现

当两条 rule 有 sample-level overlap，尤其是不同 target 的 rule overlap 时，应使用这个 category。

### 它会看哪些证据

- pair_intersection_count。
- jaccard_overlap。
- overlap_share_a / overlap_share_b。
- overlap point ids。
- 两条 rule 的 target_kind 和 target_id。
- shared features 和 threshold。

### 它应该输出什么

Overlap Merge Signal 应该说明：

- 两条 rule 共享多少点。
- Jaccard overlap 是多少。
- overlap 占每条 rule 的比例是多少。
- shared points 应该如何 label。
- 如果 shared points 的用户 label 一致，可能支持 merge 或 shared-boundary review。
- 如果 shared points 的用户 label 不一致，应该保留 separate clusters，或考虑 new cluster。

### 用户应该怎么用

用户应该优先 label overlap points，而不是先 label 两条 rule 的所有点。overlap points 是最能判断 merge 是否合理的证据。

### 典型例子

如果 rule_cluster_2 和 rule_cluster_3 共享 12 个点，Jaccard = 0.30，而且这些点在 raw feature 上非常接近，系统可以建议 label 这 12 个点。  
如果用户给这些点相同标签，可以支持 merge review；如果用户给出混合标签，说明这里可能是边界或新 cluster。

### 注意事项

如果 `intersection_count = 0`，系统必须明确说明没有 sample-level overlap。此时不能建议 merge，只能建议做 Boundary Review 或 Label Priority。

## 7. Category 4: Split Or New Cluster Signal

### 它回答什么

Split Or New Cluster Signal 负责回答：“当前某个 cluster 是否可能由多个不同子区域组成，或者是否需要创建新 cluster？”

### 什么时候出现

当同一个 target 需要多条 disjoint rules 才能解释，或者 rule coverage 低但 purity 高，或者某个 cluster 内部出现多个分离区域时，应使用这个 category。

### 它会看哪些证据

- same_target_disjoint_regions。
- coverage 低但 purity 高的 rule。
- 同一个 target 的多条 rule。
- boundary gaps 和 shared features。
- exception groups。
- candidate point groups。

### 它应该输出什么

Split Or New Cluster Signal 应该说明：

- 哪个 cluster 可能包含多个子区域。
- 哪几条 rule 分别覆盖这些子区域。
- 每个子区域的 support、coverage、purity。
- 应该分别 label 哪些 representative points。
- 什么样的 label 结果支持 split。
- 什么样的 label 结果支持 new cluster。

### 用户应该怎么用

用户应该跨子区域 label，而不是只 label 一个 rule 里的点。如果多个 disjoint regions 得到不同用户标签，split 或 new cluster 才有依据。

### 典型例子

某个 cluster 需要三条 rule 才能覆盖，而每条 rule 的 feature threshold 都不同。系统可能建议用户分别 label 每条 rule 的 representative points。  
如果三组点的标签明显不同，可能需要 split；如果其中一组标签不属于任何现有 cluster，可能需要 new cluster。

### 注意事项

Split 和 new cluster 都是高成本操作。没有足够用户 label 前，系统只能把它们作为 hypothesis，而不是直接执行。

## 8. Category 5: Anomaly Label Review

### 它回答什么

Anomaly Label Review 负责回答：“当前 outlier rule 解释的点是真 anomaly，还是只是某个 cluster 的边界点或正常成员？”

### 什么时候出现

当 rule_set 中存在 anomaly rules，或者 anomaly rule 与 cluster rule 接近、重叠、support 很低、coverage 很低时，应使用这个 category。

### 它会看哪些证据

- anomaly rule support。
- anomaly rule coverage。
- anomaly point ids。
- cluster-anomaly overlap 或 boundary relation。
- outlier score range。
- matched anomaly points。

### 它应该输出什么

Anomaly Label Review 应该说明：

- 哪些 anomaly points 最需要用户确认。
- 它们由哪条 anomaly rule 匹配。
- 这条 rule 的 support / coverage / purity。
- 如果用户认为它们是正常点，可能需要 mark normal 或调整 outlier interpretation。
- 如果用户确认它们是真 anomaly，可以强化当前 anomaly explanation。

### 用户应该怎么用

用户应该把这些点标成 true anomaly、normal member，或指定它们属于哪个 cluster。这个标签会帮助后续判断 outlier detection 是否过敏或不足。

### 典型例子

如果某条 anomaly rule 只匹配 1 个点，coverage = 0.06，系统通常不会把它当成稳定规律，而会建议用户直接确认这个点是否真异常。

### 注意事项

Anomaly Label Review 不会自动取消 outlier flag。它只提出需要人工确认的 outlier candidates。

## 9. Category 6: Exception Relabel Review

### 它回答什么

Exception Relabel Review 负责回答：“哪些被 rule 匹配但不符合当前 target 的点，最值得重新 label？”

### 什么时候出现

当 rule 有 `exception_point_ids`，或者 purity 较低，或者 rule 匹配区域混入了其他 target 的点时，应使用这个 category。

### 它会看哪些证据

- exception_point_ids。
- exception_count。
- exception_rate。
- rule purity。
- exception points 所在 rule 的 feature thresholds。
- exception points 是否集中在某个子区域。

### 它应该输出什么

Exception Relabel Review 应该说明：

- 哪些 exception points 需要 label。
- 它们违反了哪条 rule 的 target。
- exception count / exception rate 是多少。
- 如果这些 exception points 得到相同用户标签，可能说明需要 relabel、boundary fix，或 new cluster。

### 用户应该怎么用

用户应该先 label exception points，因为它们是当前规则和当前 SSDBCODI 输出最直接的矛盾点。

### 典型例子

如果 rule_cluster_2 匹配了 20 个点，其中 4 个点当前属于 cluster_3，那么这些 4 个 exception points 就是优先 relabel candidates。  
如果用户把它们都标成同一新类别，可能说明它们不是简单误差，而是新的局部结构。

### 注意事项

当前 wine.mat 默认配置里可能没有 exception points。此时系统应该显示 warning，例如 `no_exception_points_in_rules`，而不是强行编造 relabel 建议。

## 10. Category 7: Feature Label Strategy

### 它回答什么

Feature Label Strategy 负责回答：“用户 label 时应该看哪些 raw feature threshold，而不是只看 projection 位置？”

### 什么时候出现

当某些 raw features 在 rule 中反复出现，或者某个 threshold 对 cluster / anomaly 切分很关键时，应使用这个 category。

### 它会看哪些证据

- feature_usage。
- rule conditions。
- repeated thresholds。
- strongest rule 的 feature path。
- support、coverage、purity。
- candidate points。

### 它应该输出什么

Feature Label Strategy 应该说明：

- 最重要的 raw feature 是哪些。
- 哪些 threshold 频繁出现。
- 用户 label 点时应该检查哪些 feature value。
- 为什么不能只依赖 projection x/y。
- 哪些 points 可以作为 feature-based label 样本。

### 用户应该怎么用

用户在 label 候选点时，不应该只看 2D scatterplot 上的位置。应该同时查看 raw feature，例如 `proline`、`magnesium`、`flavanoids` 等，并判断这些 raw values 是否符合 domain intuition。

### 典型例子

如果当前 wine rule 中 `proline` 使用 9 次，`magnesium` 使用 2 次，Feature Label Strategy 会建议用户 label 时重点看这两个 feature，而不是只看 MDS projection。

### 注意事项

Feature Label Strategy 不表示这些 feature 一定是因果特征。它只说明这些 feature 在当前 surrogate rules 中最能解释 SSDBCODI 输出。

## 11. Category 8: Rule Confidence Audit

### 它回答什么

Rule Confidence Audit 负责回答：“这组 rule 的可信度够不够支持下一步 refinement？”

### 什么时候出现

当用户准备根据 rule 做 merge / split / anomaly relabel 前，应该先做 Rule Confidence Audit。它也适合用于检查低 purity、低 coverage、高 exception、过宽或过窄的 rule。

### 它会看哪些证据

- support_count。
- coverage。
- purity。
- rule_confidence_score。
- exception_count / exception_rate。
- quality_warnings。
- condition_count。
- matched point previews。

### 它应该输出什么

Rule Confidence Audit 应该说明：

- 哪些 rule 是稳定解释。
- 哪些 rule 只是弱证据。
- 哪些 rule 因为低 support 或低 coverage 不适合直接指导 merge / split。
- 哪些 rule 需要先 label matched sample 或 exception points。

### 用户应该怎么用

用户应该把它当成 refinement 前的安全检查。如果 audit 结果显示 rule 质量不稳定，就应该先 label，而不是直接调整 cluster。

### 典型例子

如果某条 rule purity = 1.00 但 coverage = 0.06，它虽然很干净，但只解释了很小一部分 target。它适合做局部 inspection，不适合作为整个 cluster 的定义。

### 注意事项

高 confidence 不代表可以跳过人工 label。它只代表该 rule 在当前 SSDBCODI 输出下比较一致。真正的语义正确性仍然来自用户 label。

## 12. 如何选择 category

简单规则如下：

- 不知道从哪里开始：用 `label_priority`。
- 怀疑两个 cluster 边界不稳：用 `boundary_review`。
- 两条 rule 共享 matched points：用 `overlap_merge_signal`。
- 一个 cluster 似乎由多个分离区域组成：用 `split_or_new_cluster_signal`。
- 想检查 outliers：用 `anomaly_label_review`。
- 出现 exception points：用 `exception_relabel_review`。
- 想知道 label 时看哪些 raw features：用 `feature_label_strategy`。
- 准备根据 rule 做任何 refinement 前：用 `rule_confidence_audit`。

## 13. 最重要的使用原则

1. Rule interpretation 只给 label guidance，不直接改 state。
2. Decision tree 是 surrogate rule extractor，不是新的 clustering algorithm。
3. Merge / split / new cluster 必须由用户 labels 支持。
4. Sample-level overlap 为 0 时，不能直接建议 merge。
5. Low coverage 或 low support 的 rule 可以很有用，但通常只适合局部 inspection。
6. Exception points 通常比普通 matched points 更值得优先 label。
7. Projection 只能辅助观察，最终解释应回到 raw feature thresholds。
