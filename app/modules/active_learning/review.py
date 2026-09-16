"""Six separate review queues from Dashboard_Active_Learning_LaTeX.

No LLM, hidden labels, projection distances, or cross-category aggregate score.
Scores and percentile pools are frozen in the round, before any UI filtering.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, exp

import numpy as np

from app.modules.ssdbcodi.algorithm import (
    core_distances,
    expansion_reachability,
    pairwise_euclidean,
    reachability_matrix,
)

REVIEW_VERSION = "six_scores_expansion_v1"
CATEGORIES = (
    {
        "id": "cluster_assignment_conflict",
        "name": "Cluster Assignment Conflict",
        "short": "Conflict",
        "reason": "Nearby normal points disagree with this point's group.",
        "question": "What semantic type does this point belong to?",
        "formula": "G1 = disagreeing normal neighbors / normal neighbors examined",
    },
    {
        "id": "label_coverage_gap",
        "name": "Label Coverage Gap",
        "short": "Coverage",
        "reason": "This region is far from human semantic labels.",
        "question": "Provide a semantic class to describe this region.",
        "formula": "G2 = min reachability distance to another semantic label; support radius <= a90",
    },
    {
        "id": "weak_group_reachability",
        "name": "Weak Group Reachability",
        "short": "Reach",
        "reason": "This point has weak paths to human-supported normal examples.",
        "question": "Is this a valid semantic type, or an outlier?",
        "formula": "G3 = 1 - exp(-min full expansion-path bottleneck to another normal reference)",
    },
    {
        "id": "local_sparsity",
        "name": "Local Sparsity",
        "short": "Sparse",
        "reason": "This point has little nearby data support.",
        "question": "Is this a valid rare case (Normal), an Outlier, or Unsure?",
        "formula": "G4 = 1 - exp(-mean reachability distance to MinPts nearest other records)",
    },
    {
        "id": "known_outlier_similarity",
        "name": "Known-Outlier Similarity",
        "short": "Known",
        "reason": "This point resembles an outlier you confirmed.",
        "question": "Does the confirmed outlier's issue also occur here? Normal, Outlier, or Unsure?",
        "formula": "G5 = exp(-nearest other confirmed-outlier distance); recommend only inside its local radius",
    },
    {
        "id": "model_instability",
        "name": "Model Instability",
        "short": "Unstable",
        "reason": "Nearby SSDBCODI settings disagree about this point.",
        "question": "Confirm Normal/Outlier when status flips; otherwise provide a semantic type.",
        "formula": "G6 = mean pairwise (1 - Jaccard overlap of normal-cluster member sets); two outliers agree",
    },
)
CATEGORY_IDS = tuple(item["id"] for item in CATEGORIES)


def midrank_percentiles(values):
    """Exact ties share a midpoint rank, including the candidate itself."""
    ranked = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result = {}
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        percentile = 100.0 * (start + 0.5 * (end - start)) / len(ranked)
        for point_id, _ in ranked[start:end]:
            result[point_id] = percentile
        start = end
    return result


def member_sets(analysis, point_ids):
    """Reject incomplete outputs; cluster names have no role in comparisons."""
    groups = {}
    assignments = {}
    for item in analysis.cluster_result.assignments:
        if item.cluster_id == "cluster_unassigned":
            continue
        assignments[item.point_id] = item.cluster_id
        groups.setdefault(item.cluster_id, set()).add(item.point_id)
    outliers = set(analysis.outlier_result.outlier_point_ids)
    if (set(assignments) | outliers) != set(point_ids) or set(assignments) & outliers:
        return None
    frozen_groups = {group: frozenset(members) for group, members in groups.items()}
    return {
        pid: frozenset() if pid in outliers else frozen_groups[assignments[pid]]
        for pid in point_ids
    }


def instability_scores(runs, point_ids):
    members = [member_sets(analysis, point_ids) for analysis in runs]
    if len(members) < 2 or any(item is None for item in members):
        return {}
    pairs = list(combinations(members, 2))
    return {
        pid: sum(
            1 - len(left[pid] & right[pid]) / len(left[pid] | right[pid])
            if left[pid] | right[pid]
            else 0.0
            for left, right in pairs
        )
        / len(pairs)
        for pid in point_ids
    }


def choose_batch(rows, category, limit):
    """Stable, single-category order; one representative per exact feature row."""
    ranked = sorted(
        (row for row in rows if row["scores"][category]["recommendable"]),
        key=lambda row: (-row["scores"][category]["percentile"], row["point_id"]),
    )
    seen = set()
    selected = []
    for row in ranked:
        signature = row["duplicate_key"]
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(row["point_id"])
        if len(selected) >= limit:
            break
    return selected


def build_review(
    feature_matrix,
    analysis,
    events,
    runs,
    *,
    min_pts=3,
    statuses=None,
    rereview_reasons=None,
    batch_size=4,
    threshold=75.0,
    budget_reached=False,
    run_errors=None,
):
    point_ids = feature_matrix.point_ids
    index_by_id = {pid: index for index, pid in enumerate(point_ids)}
    count = len(point_ids)
    matrix = np.asarray(feature_matrix.values, dtype=float)
    distances = pairwise_euclidean(matrix)
    radii = core_distances(distances, min_pts)
    reach = reachability_matrix(distances, radii)
    statuses = statuses or {}
    eligible = {
        pid
        for pid in point_ids
        if statuses.get(pid, "unlabeled") in {"unlabeled", "re_review"}
    }
    support_cutoff = (
        float(
            sorted(radii[index_by_id[pid]] for pid in eligible)[
                ceil(0.9 * len(eligible)) - 1
            ]
        )
        if eligible
        else None
    )
    semantic = {e.point_id for e in events if e.label_dimension == "semantic_class"}
    confirmed_outliers = {
        e.point_id
        for e in events
        if e.label_dimension == "outlier_status" and e.label_value is True
    }
    normal = (
        semantic
        | {
            e.point_id
            for e in events
            if e.label_dimension == "outlier_status" and e.label_value is False
        }
    ) - confirmed_outliers
    assignments = {
        item.point_id: item.cluster_id
        for item in analysis.cluster_result.assignments
        if item.cluster_id != "cluster_unassigned"
    }
    predicted_outliers = set(analysis.outlier_result.outlier_point_ids)
    normal_ids = set(assignments) - predicted_outliers
    tree_payload = analysis.diagnostics.get("expansion_tree", ())
    tree = tuple(
        (index_by_id[e["from"]], index_by_id[e["to"]], e["distance"])
        for e in tree_payload
    )
    barriers, origins = (
        expansion_reachability(
            count, tree, [index_by_id[pid] for pid in sorted(normal)], exclude_self=True
        )
        if len(tree) == count - 1
        else (np.full(count, np.inf), np.full(count, -1))
    )
    instability = (
        instability_scores(list(runs.values()), point_ids) if not run_errors else {}
    )
    run_members = {
        setting: member_sets(result, point_ids) for setting, result in runs.items()
    }
    run_groups = {
        setting: {a.point_id: a.cluster_id for a in result.cluster_result.assignments}
        for setting, result in runs.items()
    }
    duplicate_ids = {
        tuple(row): index for index, row in enumerate(feature_matrix.values)
    }
    rows = []
    for pid in point_ids:
        index = index_by_id[pid]
        neighbors = sorted(
            (other for other in point_ids if other != pid),
            key=lambda other: (reach[index, index_by_id[other]], other),
        )
        nearby = neighbors[:min_pts]
        scores = {}

        def add(category, raw=None, unavailable="", comparison=(), **evidence):
            scores[category] = {
                "raw": raw,
                "percentile": None,
                "recommendable": False,
                "unavailable_reason": unavailable,
                "ineligibility_reason": "",
                "comparison_point_ids": list(comparison),
                "evidence": evidence,
            }

        comparisons = [other for other in neighbors if other in normal_ids][:min_pts]
        if pid not in normal_ids:
            add(CATEGORY_IDS[0], unavailable="Current result is outlier or unassigned.")
        elif not comparisons:
            add(CATEGORY_IDS[0], unavailable="No normal comparison neighbor exists.")
        else:
            disagree = sum(
                assignments[other] != assignments[pid] for other in comparisons
            )
            add(
                CATEGORY_IDS[0],
                disagree / len(comparisons),
                comparison=comparisons,
                disagreeing=disagree,
                examined=len(comparisons),
                neighbors=[
                    {
                        "point_id": other,
                        "group": assignments[other],
                        "reachability_distance": float(
                            reach[index, index_by_id[other]]
                        ),
                    }
                    for other in comparisons
                ],
            )

        references = sorted(
            semantic - {pid},
            key=lambda other: (reach[index, index_by_id[other]], other),
        )
        if not references:
            add(CATEGORY_IDS[1], unavailable="No other human semantic label exists.")
        elif support_cutoff is None or radii[index] > support_cutoff:
            add(
                CATEGORY_IDS[1],
                unavailable="Local support check fails.",
                support_radius=float(radii[index]),
                support_cutoff=support_cutoff,
            )
        else:
            reference = references[0]
            add(
                CATEGORY_IDS[1],
                float(reach[index, index_by_id[reference]]),
                comparison=list(dict.fromkeys([reference, *nearby])),
                nearest_semantic_reference=reference,
                support_radius=float(radii[index]),
                support_cutoff=support_cutoff,
            )

        if not np.isfinite(barriers[index]):
            add(
                CATEGORY_IDS[2],
                unavailable="No other human normal reference or complete expansion path exists.",
            )
        else:
            reference = point_ids[int(origins[index])]
            add(
                CATEGORY_IDS[2],
                1 - exp(-float(barriers[index])),
                comparison=[reference],
                normal_reference=reference,
                path_barrier=float(barriers[index]),
                path_source="recorded_expansion_tree",
            )

        if count - 1 < min_pts:
            add(CATEGORY_IDS[3], unavailable="Fewer than MinPts other records exist.")
        else:
            mean_distance = float(
                np.mean([reach[index, index_by_id[other]] for other in nearby])
            )
            add(
                CATEGORY_IDS[3],
                1 - exp(-mean_distance),
                comparison=nearby,
                mean_reachability=mean_distance,
                neighbor_distances=[
                    float(reach[index, index_by_id[other]]) for other in nearby
                ],
            )

        references = sorted(
            confirmed_outliers - {pid},
            key=lambda other: (distances[index, index_by_id[other]], other),
        )
        if not references:
            add(CATEGORY_IDS[4], unavailable="No other human-confirmed outlier exists.")
        else:
            reference = references[0]
            distance = float(distances[index, index_by_id[reference]])
            radius = float(radii[index_by_id[reference]])
            add(
                CATEGORY_IDS[4],
                exp(-distance),
                comparison=[reference],
                confirmed_outlier=reference,
                distance=distance,
                local_radius=radius,
                local_match=distance <= radius,
            )

        if pid not in instability:
            add(
                CATEGORY_IDS[5],
                unavailable="At least two complete nearby-setting runs are required.",
                run_errors=run_errors or {},
            )
        else:
            alternatives = [
                {
                    "min_pts": setting,
                    "group": run_groups[setting].get(pid, "outlier"),
                    "member_count": len(members[pid]),
                    "is_outlier": not bool(members[pid]),
                }
                for setting, members in run_members.items()
            ]
            comparison = sorted(
                set().union(*(members[pid] for members in run_members.values())) - {pid}
            )[:min_pts]
            add(
                CATEGORY_IDS[5],
                instability[pid],
                comparison=comparison,
                runs=alternatives,
                status_flips=len({item["is_outlier"] for item in alternatives}) > 1,
            )
        rows.append(
            {
                "point_id": pid,
                "group": "outlier"
                if pid in predicted_outliers
                else assignments.get(pid, "unassigned"),
                "review_status": statuses.get(pid, "unlabeled"),
                "eligible": pid in eligible,
                "rereview_reason": (rereview_reasons or {}).get(pid, ""),
                "duplicate_key": duplicate_ids[tuple(feature_matrix.values[index])],
                "scores": scores,
            }
        )

    categories = []
    for definition in CATEGORIES:
        category = definition["id"]
        pool = {
            row["point_id"]: row["scores"][category]["raw"]
            for row in rows
            if row["eligible"] and row["scores"][category]["raw"] is not None
        }
        percentiles = midrank_percentiles(pool)
        for row in rows:
            score = row["scores"][category]
            score["percentile"] = percentiles.get(row["point_id"])
            reason = score["unavailable_reason"]
            if not reason and not row["eligible"]:
                reason = (
                    "Already reviewed or deferred; not in this round's eligible pool."
                )
            if not reason and budget_reached:
                reason = "The session label budget is complete."
            if (
                not reason
                and category
                in (CATEGORY_IDS[0], CATEGORY_IDS[2], CATEGORY_IDS[3], CATEGORY_IDS[5])
                and score["raw"] <= 0
            ):
                reason = "Zero raw score supplies no review reason."
            if (
                not reason
                and category != CATEGORY_IDS[0]
                and score["percentile"] < threshold
            ):
                reason = f"Priority is below the {threshold:g} percentile recommendation threshold."
            if (
                not reason
                and category == CATEGORY_IDS[4]
                and not score["evidence"]["local_match"]
            ):
                reason = (
                    "The nearest confirmed outlier is outside its local-match radius."
                )
            score["ineligibility_reason"] = reason
            score["recommendable"] = not bool(reason)
        categories.append(
            {
                **definition,
                "pool_size": len(pool),
                "all_tied": bool(pool) and len(set(pool.values())) == 1,
                "recommended_point_ids": choose_batch(rows, category, batch_size),
            }
        )
    return {
        "version": REVIEW_VERSION,
        "categories": categories,
        "rows": rows,
        "min_pts": min_pts,
        "percentile_threshold": threshold,
        "coverage_support_quantile": 0.9,
        "expansion_tree": list(tree_payload),
        "instability_settings": list(runs),
        "instability_groups": {
            str(setting): {
                group: sorted(
                    pid
                    for pid, assigned in run_groups[setting].items()
                    if assigned == group
                )
                for group in sorted(set(run_groups[setting].values()))
            }
            for setting, members in run_members.items()
            if members is not None
        },
        "run_errors": run_errors or {},
    }
