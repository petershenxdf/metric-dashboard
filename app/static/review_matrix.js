/* Pure view operations never recalculate the round's percentile pools. */
(function (root) {
  'use strict';
  const byId = (a, b) => a < b ? -1 : a > b ? 1 : 0;
  function band(value) {
    if (value === null || value === undefined) return 'band-na';
    return 'band-' + (value < 25 ? 0 : value < 50 ? 1 : value < 75 ? 2 : value < 90 ? 3 : 4);
  }
  function sortRows(rows, category, descending = true) {
    return [...rows].sort((a, b) => {
      const left = a.scores[category].percentile, right = b.scores[category].percentile;
      if (left === null && right !== null) return 1;
      if (right === null && left !== null) return -1;
      return (left === null ? 0 : (left - right) * (descending ? -1 : 1)) || byId(a.point_id, b.point_id);
    });
  }
  function filterRows(rows, filters) {
    return rows.filter(row => {
      if (filters.status !== 'all' && row.review_status !== filters.status) return false;
      if (filters.group && row.group !== filters.group) return false;
      if (filters.search && !row.point_id.toLowerCase().includes(filters.search.toLowerCase())) return false;
      if (filters.recommendable && !row.scores[filters.category].recommendable) return false;
      const checks = filters.conditions.map(condition => {
        const value = row.scores[condition.category].percentile;
        return condition.kind === 'na' ? value === null : value !== null && value >= condition.value;
      });
      return !checks.length || (filters.mode === 'any' ? checks.some(Boolean) : checks.every(Boolean));
    });
  }
  function batch(rows, category, limit) {
    const seen = new Set(), result = [];
    for (const row of sortRows(rows, category)) {
      if (!row.scores[category].recommendable || seen.has(row.duplicate_key)) continue;
      seen.add(row.duplicate_key);
      result.push(row.point_id);
      if (result.length >= limit) break;
    }
    return result;
  }
  function scorePresentation(score, row) {
    if (score.percentile !== null) return {label: String(Math.round(score.percentile)), reason: '', className: band(score.percentile)};
    if (score.raw !== null && !row.eligible) return {
      label: row.review_status === 'deferred' ? 'Deferred' : 'Reviewed',
      reason: 'Outside ranking pool', className: 'band-reviewed'};
    const reason = score.unavailable_reason || score.ineligibility_reason || 'Evidence unavailable';
    let short = reason;
    if (reason.includes('semantic label')) short = 'Needs semantic label';
    else if (reason.includes('normal reference')) short = 'Needs normal reference / path';
    else if (reason.includes('confirmed outlier')) short = 'Needs confirmed outlier';
    else if (reason.includes('outlier or unassigned')) short = 'Not applicable';
    else if (reason.includes('support check')) short = 'Insufficient local support';
    else if (reason.includes('nearby-setting runs')) short = 'Incomplete model runs';
    return {label: 'NA', reason: short, className: 'band-na'};
  }
  function scoreExplanation(category, score) {
    if (score.raw === null) return score.unavailable_reason;
    const evidence = score.evidence;
    if (category === 'cluster_assignment_conflict') return evidence.disagreeing + ' of ' + evidence.examined + ' nearest normal comparison points are in another cluster.';
    if (category === 'model_instability') return score.raw === 0
      ? 'Nearby SSDBCODI settings agree on this point’s status and normal-group membership.'
      : 'Nearby SSDBCODI settings disagree about this point.';
    if (category === 'label_coverage_gap') return 'Nearest human semantic reference: ' + evidence.nearest_semantic_reference + '. Larger distance means less label coverage.';
    if (category === 'weak_group_reachability') return 'Best recorded path to normal reference ' + evidence.normal_reference + '; a larger path barrier means weaker normal support.';
    if (category === 'local_sparsity') return 'Local support is measured using the nearest comparison records; larger raw scores mean sparser surroundings.';
    return 'Nearest confirmed outlier: ' + evidence.confirmed_outlier + '. ' + (evidence.local_match ? 'Inside its local-match radius.' : 'Outside its local-match radius; not recommended by this category.');
  }
  root.ReviewMatrix = {band, sortRows, filterRows, batch, scorePresentation, scoreExplanation};
  if (typeof document === 'undefined') return;
  const $ = id => document.getElementById(id);
  const stateElement = $('review-state');
  if (!stateElement) return;
  const state = JSON.parse(stateElement.textContent);
  const rows = state.review.rows || [];
  const categories = state.review.categories || [];
  const definitions = new Map(categories.map(item => [item.id, item]));
  const rowById = new Map(rows.map(row => [row.point_id, row]));
  const pointById = new Map(state.plot_points.map(point => [point.point_id, point]));
  const plotNodes = new Map([...document.querySelectorAll('.review-plot-point')].map(node => [node.dataset.pointId, node]));
  const query = new URLSearchParams(location.search);
  let selected = query.get('selected') || null;
  if (!pointById.has(selected)) selected = null;
  let category = state.focus_category;
  let descending = true, conditions = [], visible = [], busy = false;
  let comparison = new Set(), recommended = new Set();
  const checked = new Set();
  const apiBase = '/api/active-learning/sessions/' + encodeURIComponent(state.session.session_id);
  const roundBase = apiBase + '/rounds/' + encodeURIComponent(state.round.round_id);

  function labelTargets() {
    return $('label-scope').value === 'batch' ? [...checked] : selected ? [selected] : [];
  }
  function saveView() {
    const url = new URL(location.href);
    const filters = currentFilters();
    url.searchParams.set('view', JSON.stringify({...filters, descending}));
    if (selected) url.searchParams.set('selected', selected);
    url.searchParams.set('focus_category', category);
    if (url.href !== location.href) history.replaceState(null, '', url.href);
  }
  function restoreView() {
    let view;
    try { view = JSON.parse(query.get('view')); } catch (_) { return; }
    if (!view || typeof view !== 'object') return;
    if (['unlabeled', 're_review', 'labeled', 'deferred', 'all'].includes(view.status)) $('status-filter').value = view.status;
    if (typeof view.group === 'string') {
      if (view.group && ![...$('group-filter').options].some(option => option.value === view.group)) {
        const option = element('option', view.group + ' (not in this round)'); option.value = view.group; $('group-filter').append(option);
      }
      $('group-filter').value = view.group;
    }
    if (typeof view.search === 'string') $('point-search').value = view.search;
    $('recommendable-only').checked = view.recommendable === true;
    if (definitions.has(view.category)) $('sort-category').value = view.category;
    $('condition-mode').value = view.mode === 'any' ? 'any' : 'all';
    descending = view.descending !== false;
    if (Array.isArray(view.conditions)) conditions = view.conditions.filter(c => c && definitions.has(c.category) &&
      (c.kind === 'na' || (c.kind === 'minimum' && typeof c.value === 'number' && Number.isFinite(c.value) && c.value >= 0 && c.value <= 100)));
  }
  function showPoolGuidance() {
    const semantic = new Set(state.active_labels.filter(e => e.label_dimension === 'semantic_class').map(e => e.point_id));
    const outliers = new Set(state.active_labels.filter(e => e.label_dimension === 'outlier_status' && e.label_value === true).map(e => e.point_id));
    const normals = new Set([...semantic, ...state.active_labels.filter(e => e.label_dimension === 'outlier_status' && e.label_value === false).map(e => e.point_id)].filter(pid => !outliers.has(pid)));
    const missing = new Map([
      ['label_coverage_gap', !semantic.size ? 'Needs a human semantic label' : ''],
      ['weak_group_reachability', !normals.size ? 'Needs a human normal reference' : ''],
      ['known_outlier_similarity', !outliers.size ? 'Needs a confirmed outlier' : '']]);
    for (const node of document.querySelectorAll('[data-pool-guidance]')) {
      const definition = definitions.get(node.dataset.poolGuidance);
      node.textContent = missing.get(definition.id) || (definition.pool_size ? '' : 'No eligible records with sufficient evidence');
    }
    $('review-guidance').textContent = [...missing.values()].some(Boolean)
      ? 'Some columns are waiting for human references. Start with Conflict, Sparse or Unstable: add a semantic type to enable Coverage, or confirm Normal to enable Reach. Known needs a genuinely confirmed Outlier — do not invent labels to fill the matrix.'
      : 'Human references are available. NA tiles explain missing or inapplicable evidence; Reviewed / Deferred records are outside the ranking pool. Scores are percentiles, not anomaly probabilities.';
  }

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function button(text, action) {
    const node = element('button', text);
    node.type = 'button'; node.addEventListener('click', action);
    return node;
  }
  function message(text, error = false) {
    $('action-message').textContent = text;
    $('action-message').classList.toggle('error', error);
  }
  function currentFilters() {
    return {status: $('status-filter').value, group: $('group-filter').value,
            search: $('point-search').value.trim(), recommendable: $('recommendable-only').checked,
            category: $('sort-category').value, conditions, mode: $('condition-mode').value};
  }
  function updatePlot() {
    for (const [pid, node] of plotNodes) {
      node.classList.toggle('selected', pid === selected);
      node.classList.toggle('comparison', comparison.has(pid));
      node.classList.toggle('recommended', recommended.has(pid));
      node.classList.toggle('batch-selected', checked.has(pid));
      node.setAttribute('aria-pressed', String(pid === selected));
    }
  }
  function updateSelection() {
    for (const rowNode of document.querySelectorAll('#matrix-rows tr')) {
      rowNode.classList.toggle('selected', rowNode.dataset.pointId === selected);
      rowNode.classList.toggle('batch-selected', checked.has(rowNode.dataset.pointId));
    }
    $('selected-hidden').textContent = selected && rows.length && !visible.some(row => row.point_id === selected)
      ? 'Selected point ' + selected + ' is hidden by the current filters.' : '';
    if ($('matrix-selection')) {
      $('matrix-selection').textContent = $('selected-hidden').textContent || (selected ? 'Selected: ' + selected : '');
      $('show-selected').hidden = !selected;
    }
    if ($('label-targets')) {
      const targets = labelTargets();
      const hidden = [...checked].filter(pid => !visible.some(row => row.point_id === pid)).length;
      $('label-targets').textContent = targets.length ? ($('label-scope').value === 'batch' ? 'BATCH — ' : 'SINGLE — ') + 'Label ' + targets.length + ' record(s): ' + targets.join(', ') +
        (hidden ? ' (' + hidden + ' checked record(s) hidden by filters)' : '') : ($('label-scope').value === 'batch' ? 'No batch targets. Check records or switch to Single.' : 'Select a point to label.');
      $('label-scope-warning').textContent = $('label-scope').value === 'batch' && selected
        ? 'Details show ' + selected + '. Submission applies to the checked batch, not automatically to the focused point.' : '';
      document.querySelectorAll('[data-label-action]').forEach(control => { control.disabled = busy || !targets.length; });
    }
    updatePlot();
  }
  function expansionPath(target, source) {
    const adjacency = new Map();
    for (const edge of state.review.expansion_tree || []) {
      for (const [a, b] of [[edge.from, edge.to], [edge.to, edge.from]]) {
        if (!adjacency.has(a)) adjacency.set(a, []);
        adjacency.get(a).push({point: b, distance: edge.distance});
      }
    }
    const stack = [source], parents = new Map([[source, null]]), weights = new Map();
    while (stack.length) {
      const point = stack.pop();
      if (point === target) break;
      for (const edge of adjacency.get(point) || []) {
        if (parents.has(edge.point)) continue;
        parents.set(edge.point, point); weights.set(edge.point, edge.distance); stack.push(edge.point);
      }
    }
    if (!parents.has(target)) return [];
    const path = [];
    for (let point = target; parents.get(point) !== null; point = parents.get(point)) {
      path.unshift({from: parents.get(point), to: point, reachability_distance: weights.get(point)});
    }
    return path;
  }
  function showDetails() {
    const panel = $('point-details'); panel.replaceChildren(); comparison = new Set();
    if (!selected) return;
    const point = pointById.get(selected), row = rowById.get(selected);
    $('selected-id').textContent = selected + ' · Current model result: ' + (point.is_outlier ? 'outlier' : point.cluster_id);
    if (row) {
      const score = row.scores[category], definition = definitions.get(category);
      panel.append(element('h3', definition.name));
      const sentence = scoreExplanation(category, score);
      panel.append(element('p', sentence));
      panel.append(element('p', score.percentile === null ? (score.raw === null ? 'Priority: NA — ' + score.unavailable_reason :
        'Outside the ranking pool (' + row.review_status + '). Raw score: ' + score.raw.toPrecision(8)) :
        'Priority percentile: ' + score.percentile.toFixed(6) + ' · Raw score: ' + score.raw.toPrecision(8)));
      if (definition.all_tied) panel.append(element('p', 'All tied in this category’s eligible pool — no useful ordering between these equal scores.'));
      panel.append(element('p', score.recommendable ? 'Meets this category’s recommendation conditions.' : score.ineligibility_reason));
      if (row.rereview_reason) panel.append(element('p', 'Re-review: ' + row.rereview_reason));
      panel.append(element('p', definition.question));
      comparison = new Set(score.comparison_point_ids);
      const evidence = {...score.evidence};
      if (evidence.normal_reference) {
        evidence.full_expansion_path = expansionPath(selected, evidence.normal_reference);
        for (const edge of evidence.full_expansion_path) { comparison.add(edge.from); comparison.add(edge.to); }
        comparison.delete(selected);
      }
      if (category === 'model_instability' && score.raw !== null) {
        for (const run of evidence.runs) {
          const members = run.is_outlier ? [] : ((state.review.instability_groups || {})[String(run.min_pts)] || {})[run.group] || [];
          panel.append(element('p', 'MinPts ' + run.min_pts + ': ' + (run.is_outlier ? 'Outlier' : 'Normal · ' + run.member_count + ' group members')));
          const details = element('details'); details.append(element('summary', 'Exact member set (MinPts ' + run.min_pts + ')'));
          details.append(element('p', members.length ? members.join(', ') : 'Empty set (outlier)'));
          panel.append(details);
        }
      }
      const technical = element('details'); technical.append(element('summary', 'Raw value, formula and evidence'));
      technical.append(element('p', definition.formula));
      technical.append(element('pre', JSON.stringify({raw_score: score.raw, priority_percentile: score.percentile, ...evidence}, null, 2)));
      panel.append(technical);
      const comparisonSection = element('details'); comparisonSection.open = true;
      comparisonSection.append(element('summary', 'Comparison records (' + comparison.size + ')'));
      const list = element('div', undefined, 'review-actions');
      for (const pid of comparison) {
        const other = pointById.get(pid);
        list.append(button(pid + ' · ' + (other.is_outlier ? 'outlier' : other.cluster_id), () => select(pid, category)));
      }
      comparisonSection.append(list); panel.append(comparisonSection);
    }
    const labels = state.active_labels.filter(label => label.point_id === selected);
    if (labels.length) panel.append(element('p', 'Saved human feedback: ' + labels.map(label =>
      label.label_dimension + ' = ' + (state.session.label_vocabulary[label.label_value] || String(label.label_value))).join('; ')));
    const features = element('details'); features.append(element('summary', 'Raw features and metadata'));
    features.append(element('pre', JSON.stringify({features: point.raw_features, metadata: point.metadata}, null, 2)));
    panel.append(features);
  }
  function select(pid, focusCategory = category) {
    selected = pid; category = focusCategory;
    showDetails(); updateSelection(); if (categories.length) saveView();
  }
  function chip(text, remove) {
    $('filter-chips').append(button(text + ' ×', () => { remove(); render(); }));
  }
  function renderChips(filters) {
    $('filter-chips').replaceChildren();
    if (filters.status !== 'all') chip('Status: ' + filters.status, () => { $('status-filter').value = 'all'; });
    if (filters.group) chip('Group: ' + filters.group, () => { $('group-filter').value = ''; });
    if (filters.search) chip('ID: ' + filters.search, () => { $('point-search').value = ''; });
    if (filters.recommendable) chip('Recommendable: ' + definitions.get(filters.category).name, () => { $('recommendable-only').checked = false; });
    conditions.forEach((condition, index) => chip(definitions.get(condition.category).short +
      (condition.kind === 'na' ? ' is NA' : ' ≥ ' + condition.value), () => { conditions.splice(index, 1); }));
  }
  function render() {
    if (!rows.length) return;
    const filters = currentFilters(), sortCategory = filters.category;
    visible = sortRows(filterRows(rows, filters), sortCategory, descending);
    recommended = new Set(batch(visible, sortCategory, state.session.config.batch_size));
    const body = $('matrix-rows'); body.replaceChildren();
    const fragment = document.createDocumentFragment();
    for (const row of visible) {
      const tr = element('tr'); tr.dataset.pointId = row.point_id;
      tr.addEventListener('click', () => select(row.point_id, sortCategory));
      const checkCell = element('td'), check = element('input'); check.type = 'checkbox'; check.checked = checked.has(row.point_id);
      check.setAttribute('aria-label', 'Include ' + row.point_id + ' in label batch');
      check.addEventListener('click', event => event.stopPropagation());
      check.addEventListener('change', () => {
        check.checked ? checked.add(row.point_id) : checked.delete(row.point_id);
        $('label-scope').value = 'batch'; updateSelection();
      });
      checkCell.append(check); tr.append(checkCell);
      const idCell = element('td'); idCell.append(button(row.point_id, event => { event.stopPropagation(); select(row.point_id, sortCategory); }));
      idCell.append(element('small', row.group + ' · ' + row.review_status.replace('_', ' '))); tr.append(idCell);
      for (const definition of categories) {
        const score = row.scores[definition.id], cell = element('td');
        const presentation = scorePresentation(score, row);
        const tile = button(presentation.label, event => {
          event.stopPropagation(); select(row.point_id, definition.id);
        });
        tile.className = 'score-tile ' + presentation.className;
        if (presentation.reason) tile.append(element('small', presentation.reason, 'score-reason'));
        tile.title = definition.name + ': ' + (score.percentile === null ? (score.unavailable_reason || score.ineligibility_reason) : score.percentile.toFixed(6) + ' percentile; raw ' + score.raw);
        tile.setAttribute('aria-label', row.point_id + ' · ' + tile.title);
        cell.append(tile); tr.append(cell);
      }
      fragment.append(tr);
    }
    body.append(fragment);
    for (const heading of document.querySelectorAll('[data-category-heading]')) {
      heading.setAttribute('aria-sort', heading.dataset.categoryHeading === sortCategory ? (descending ? 'descending' : 'ascending') : 'none');
      const control = heading.querySelector('button');
      control.textContent = definitions.get(heading.dataset.categoryHeading).short + (heading.dataset.categoryHeading === sortCategory ? (descending ? ' ↓' : ' ↑') : '');
    }
    $('sort-direction').textContent = descending ? 'Highest first ↓' : 'Lowest first ↑';
    $('matrix-summary').textContent = visible.length + ' of ' + rows.length + ' shown · Sorted by ' + definitions.get(sortCategory).name +
      ' · ' + recommended.size + ' recommended in visible rows' + (recommended.size ? '' : ' — no forced recommendation');
    $('matrix-empty').hidden = visible.length !== 0;
    renderChips(filters); updateSelection(); saveView();
  }

  async function post(path, payload, label) {
    if (busy) return;
    busy = true;
    const controls = [...document.querySelectorAll('[data-label-action], [data-revert], #upgrade-round')];
    controls.forEach(control => { control.disabled = true; });
    message(label + ' — calculating the next round…');
    try {
      const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error && result.error.message || 'Request failed (' + response.status + ')');
      const url = new URL(location.href);
      url.searchParams.delete('show_interpretation'); url.searchParams.delete('provider_kind');
      if (selected) url.searchParams.set('selected', selected);
      url.searchParams.set('focus_category', category);
      location.assign(url.href);
    } catch (error) {
      message(error.message + ' Your selection is retained; refresh if the round changed.', true);
      busy = false; controls.forEach(control => { control.disabled = false; }); updateSelection();
    }
  }
  for (const [pid, node] of plotNodes) {
    node.addEventListener('click', () => select(pid));
    node.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(pid); }
    });
  }
  const legendGroups = new Map();
  for (const point of state.plot_points) if (!point.is_outlier) legendGroups.set(point.cluster_id, point.color);
  for (const [group, color] of legendGroups) {
    const label = element('span'), dot = element('i', undefined, 'cluster-dot'); dot.style.backgroundColor = color;
    label.append(dot, document.createTextNode(group)); $('review-plot-legend').append(label);
  }
  $('review-plot-legend').append(element('span', '◆ Outlier'));
  for (const control of document.querySelectorAll('[data-revert]')) control.addEventListener('click', () => {
    if (confirm('Return to this round? Later label events will be retracted from the active branch, but kept in history.')) {
      post(apiBase + '/rounds/' + encodeURIComponent(control.dataset.revert) + '/revert', {}, 'Returning to saved round');
    }
  });
  if ($('upgrade-round')) $('upgrade-round').addEventListener('click', () => post(apiBase + '/upgrade', {expected_round_id: state.round.round_id}, 'Upgrading review'));
  if (categories.length) {
    $('show-selected').addEventListener('click', () => {
      $('details-title').scrollIntoView({behavior: 'smooth', block: 'start'});
    });
    $('sort-category').value = category;
    for (const group of [...new Set(rows.map(row => row.group))].sort(byId)) {
      const option = element('option', group); option.value = group; $('group-filter').append(option);
    }
    restoreView(); showPoolGuidance();
    for (const id of ['status-filter', 'group-filter', 'recommendable-only', 'condition-mode']) $(id).addEventListener('change', render);
    $('point-search').addEventListener('input', render);
    $('sort-category').addEventListener('change', () => { category = $('sort-category').value; descending = true; showDetails(); render(); });
    $('sort-direction').addEventListener('click', () => { descending = !descending; render(); });
    for (const control of document.querySelectorAll('[data-sort]')) control.addEventListener('click', () => {
      descending = $('sort-category').value === control.dataset.sort ? !descending : true;
      category = control.dataset.sort; $('sort-category').value = category; showDetails(); render();
    });
    $('condition-kind').addEventListener('change', () => { $('condition-value').disabled = $('condition-kind').value === 'na'; });
    $('add-condition').addEventListener('click', () => {
      const value = Number($('condition-value').value), kind = $('condition-kind').value;
      if (kind !== 'na' && ($('condition-value').value === '' || !Number.isFinite(value) || value < 0 || value > 100)) {
        message('Choose a percentile from 0 to 100.', true); return;
      }
      conditions.push({category: $('condition-category').value, kind, value}); render();
    });
    $('reset-filters').addEventListener('click', () => {
      $('status-filter').value = 'unlabeled'; $('group-filter').value = ''; $('point-search').value = '';
      $('recommendable-only').checked = false; $('condition-mode').value = 'all'; conditions = []; render();
    });
    $('select-batch').addEventListener('click', () => {
      checked.clear(); for (const pid of recommended) checked.add(pid); $('label-scope').value = 'batch'; render();
    });
    function useFocusedPoint() { checked.clear(); $('label-scope').value = 'single'; render(); }
    $('clear-batch').addEventListener('click', useFocusedPoint);
    $('use-focused-point').addEventListener('click', useFocusedPoint);
    $('label-scope').addEventListener('change', () => {
      if ($('label-scope').value === 'single') checked.clear();
      render();
    });
    for (const control of document.querySelectorAll('[data-label-action]')) control.addEventListener('click', () => {
      const targets = labelTargets();
      if (!targets.length) { message('Select a point or check a batch first.', true); return; }
      const action = control.dataset.labelAction;
      const dimension = action === 'semantic' ? 'semantic_class' : action === 'uncertain' ? 'uncertain' : 'outlier_status';
      const value = action === 'semantic' ? $('semantic-label').value.trim() : action !== 'normal';
      if (action === 'semantic' && !value) { message('Enter a human-defined semantic type first.', true); return; }
      if ($('label-scope').value === 'batch' && !confirm('Apply ' + (action === 'semantic' ? 'semantic type "' + value + '"' : action) +
          ' to these ' + targets.length + ' checked records?\n' + targets.join(', ') + '\nThis does not label other focused or visible records.')) return;
      post(roundBase + '/labels', {expected_round_id: state.round.round_id, expected_label_revision: state.round.label_revision,
        category, labels: targets.map(point_id => ({point_id, label_dimension: dimension, label_value: value}))}, 'Saving human feedback');
    });
    render();
  }
  if (selected) { showDetails(); updateSelection(); }
})(typeof globalThis !== 'undefined' ? globalThis : this);
