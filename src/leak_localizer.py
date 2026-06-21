"""
leak_localizer.py — Hydraulic Sensitivity Matrix Leak Localizer
================================================================
Identifies the most likely leak location by solving a weighted least-squares
problem between observed pressure residuals and a precomputed hydraulic
sensitivity matrix S.

Physics
-------
For a leak of flow q_L [L/s] at node j, the pressure change at sensor i is:
    ΔP_i ≈ S[i,j] * q_L

where S[i,j] = ∂P_i/∂q_j [m / (L/s)] is the pressure sensitivity to demand
perturbation at node j (computed numerically in network_simulator.py).

S is always negative (any withdrawal lowers pressure everywhere downstream).

Localization criterion
----------------------
Given residual vector r = P_measured - P_predicted (negative during a leak):
    Find node j* = argmin_j ||r - S[:,j] * q_j||²

This is a 1-parameter LS problem per candidate node (q_j is the unknown leak
flow). The solution is:
    q_j_hat = S[:,j]^T * r / ||S[:,j]||²

Node with highest |q_j_hat| and smallest normalized residual is ranked first.

Limitations (documented honestly)
----------------------------------
- Linear hydraulic model: valid for small leaks (< ~20% of branch flow)
- Sensor placement matters: nodes with no nearby sensor have poor coverage
- Demand uncertainty couples with leak estimation — not separated here
- No time-domain localization (spatial only, not temporal)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class LeakLocalizer:
    """
    Sensitivity-matrix based leak localizer.

    Parameters
    ----------
    sensitivity_matrix : np.ndarray, shape (n_sensors, n_nodes)
    sensor_ids : list of str
    node_ids : list of str
    """

    def __init__(
        self,
        sensitivity_matrix: np.ndarray,
        sensor_ids: List[str],
        node_ids: List[str],
        lasso_alpha: float = 0.01,
    ):
        self.S = sensitivity_matrix
        self.sensor_ids = sensor_ids
        self.node_ids = node_ids
        self.lasso_alpha = lasso_alpha
        self._n_sensors, self._n_nodes = self.S.shape

        # Precompute column norms for efficiency
        self._col_norms = np.linalg.norm(self.S, axis=0)

    def localize(
        self,
        residuals_per_sensor: Dict[str, float],
        top_k: int = 5,
        method: str = "wls",
    ) -> List[dict]:
        """
        Rank candidate leak nodes given a snapshot of pressure residuals.

        Parameters
        ----------
        residuals_per_sensor : dict {sensor_id: residual [m H₂O]}
            Typically the median residual over a detected leak window.
        top_k : int
            Number of top candidates to return. Default: 5.
        method : str
            Localization method. Choices: "wls" or "lasso".

        Returns
        -------
        list of dict (ranked by score, best first):
            {
              'node': str,
              'estimated_leak_flow_ls': float,  # LS estimate of leak flow [L/s]
              'score': float,                    # normalized fit quality (higher = better)
              'rank': int,
            }
        """
        # Build residual vector aligned with sensor_ids order
        r = np.array([
            residuals_per_sensor.get(sid, 0.0)
            for sid in self.sensor_ids
        ])

        if method == "lasso":
            from sklearn.linear_model import Lasso
            # S is naturally negative, residuals are negative. We solve NN-Lasso using positive features/targets.
            X = -self.S
            y = -r
            lasso = Lasso(alpha=self.lasso_alpha, positive=True, fit_intercept=False, max_iter=5000)
            lasso.fit(X, y)
            q_lasso = lasso.coef_

            # Also calculate WLS scores for ranking zero-coefficient nodes
            wls_scores = []
            for j, node_id in enumerate(self.node_ids):
                s_j = self.S[:, j]
                norm_j = self._col_norms[j]
                if norm_j < 1e-9:
                    wls_scores.append((node_id, 0.0, 0.0))
                    continue
                q_wls = float(np.dot(s_j, r) / (norm_j ** 2))
                r_pred = s_j * q_wls
                ss_res = np.sum((r - r_pred) ** 2)
                ss_tot = np.sum(r ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
                score = max(r2, 0.0) if q_wls > 0 else 0.0
                wls_scores.append((node_id, q_wls, score))

            scores = []
            for j, node_id in enumerate(self.node_ids):
                q_l = float(q_lasso[j])
                q_w, w_score = wls_scores[j][1], wls_scores[j][2]
                
                # Lasso coefficients dictate the ranking.
                # Nodes with non-zero Lasso coefficients sort first.
                if q_l > 0:
                    composite = q_l
                else:
                    composite = 1e-9 * w_score

                scores.append({
                    'node': node_id,
                    'estimated_leak_flow_ls': q_l if q_l > 0 else q_w,
                    # For metrics/plotting, report original WLS fit quality score (or scaled down if zero-lasso)
                    'score': float(w_score) if q_l > 0 else float(w_score * 0.1),
                    'composite_score': composite,
                })
            scores.sort(key=lambda x: -x['composite_score'])
        else:
            # Traditional WLS
            scores = []
            for j, node_id in enumerate(self.node_ids):
                s_j = self.S[:, j]
                norm_j = self._col_norms[j]

                if norm_j < 1e-9:
                    scores.append({
                        'node': node_id,
                        'estimated_leak_flow_ls': 0.0,
                        'score': 0.0,
                    })
                    continue

                q_hat = float(np.dot(s_j, r) / (norm_j ** 2))
                r_pred = s_j * q_hat
                ss_res = np.sum((r - r_pred) ** 2)
                ss_tot = np.sum(r ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
                score = max(r2, 0.0) if q_hat > 0 else 0.0

                scores.append({
                    'node': node_id,
                    'estimated_leak_flow_ls': max(q_hat, 0.0),
                    'score': float(score),
                })
            scores.sort(key=lambda x: -x['score'])

        for rank, s in enumerate(scores[:top_k]):
            s['rank'] = rank + 1

        return scores[:top_k]

    def localize_over_window(
        self,
        timestamps_h: np.ndarray,
        residuals: Dict[str, np.ndarray],
        alarm_start_h: float,
        window_hours: float = 24.0,
        top_k: int = 5,
        method: str = "wls",
    ) -> List[dict]:
        """
        Aggregate residuals over a time window following alarm onset and localize.

        Uses median residual over the window (robust to diurnal fluctuation).

        Parameters
        ----------
        timestamps_h : np.ndarray
        residuals : dict {sensor_id: residual_array}
        alarm_start_h : float
            First alarm timestamp [hours] — start of analysis window.
        window_hours : float
            Analysis window duration [hours]. Default: 24.
        top_k : int
        method : str
            Localization method. Choices: "wls" or "lasso".

        Returns
        -------
        Ranked candidate nodes (same format as localize()).
        """
        mask = (timestamps_h >= alarm_start_h) & (timestamps_h < alarm_start_h + window_hours)
        if mask.sum() == 0:
            return []

        median_residuals = {}
        for sid, r_arr in residuals.items():
            window_vals = r_arr[mask]
            # Filter NaN (lag period)
            valid = window_vals[~np.isnan(window_vals)]
            median_residuals[sid] = float(np.median(valid)) if len(valid) > 0 else 0.0

        return self.localize(median_residuals, top_k=top_k, method=method)

    @staticmethod
    def localization_accuracy(
        candidates: List[dict],
        true_node: str | List[str],
        node_ids: List[str],
        adjacency: Optional[Dict[str, List[str]]] = None,
    ) -> dict:
        """
        Evaluate localization result against ground truth.

        Parameters
        ----------
        candidates : list of ranked candidate dicts
        true_node : str or list of str
            True node(s) representing the leak source. If a pipe is leaking,
            this can be a list containing its start and end nodes.
        node_ids : list of str
        adjacency : dict {node_id: [neighbor_ids]}, optional
            Network adjacency for hop-distance computation.

        Returns
        -------
        dict:
            'top1_correct': bool,
            'top3_correct': bool,
            'rank_of_true': int or None,
            'hop_distance': int or None (requires adjacency)
        """
        if isinstance(true_node, str):
            true_nodes = [true_node.lower()]
        else:
            true_nodes = [tn.lower() for tn in true_node]

        node_ranks = {c['node'].lower(): c['rank'] for c in candidates}
        
        ranks = [node_ranks[tn] for tn in true_nodes if tn in node_ranks]
        rank_of_true = min(ranks) if ranks else None

        top1 = candidates[0]['node'].lower() in true_nodes if candidates else False
        top3 = any(c['node'].lower() in true_nodes for c in candidates[:3]) if candidates else False

        hop_dist = None
        if adjacency is not None and candidates:
            best_guess = candidates[0]['node'].lower()
            # case-insensitive adjacency dict
            adj_lower = {k.lower(): [v.lower() for v in neighbors] for k, neighbors in adjacency.items()}
            dists = []
            for tn in true_nodes:
                d = _bfs_hop_distance(best_guess, tn, adj_lower)
                if d is not None:
                    dists.append(d)
            hop_dist = min(dists) if dists else None

        return {
            'top1_correct': top1,
            'top3_correct': top3,
            'rank_of_true': rank_of_true,
            'hop_distance': hop_dist,
        }


def _bfs_hop_distance(
    source: str,
    target: str,
    adjacency: Dict[str, List[str]],
) -> Optional[int]:
    """BFS shortest path hop count between source and target."""
    if source == target:
        return 0
    visited = {source}
    queue = [(source, 0)]
    while queue:
        node, dist = queue.pop(0)
        for neighbor in adjacency.get(node, []):
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None  # not connected


def build_adjacency_from_pipes(pipes) -> Dict[str, List[str]]:
    """Build undirected adjacency dict from pipe list (excludes reservoir)."""
    adj: Dict[str, List[str]] = {}
    for pipe in pipes:
        f, t = pipe.from_node, pipe.to_node
        if f != "R" and t != "R":
            adj.setdefault(f, []).append(t)
            adj.setdefault(t, []).append(f)
    return adj
