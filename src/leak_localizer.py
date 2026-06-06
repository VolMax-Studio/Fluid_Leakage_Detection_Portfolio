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
    ):
        self.S = sensitivity_matrix
        self.sensor_ids = sensor_ids
        self.node_ids = node_ids
        self._n_sensors, self._n_nodes = self.S.shape

        # Precompute column norms for efficiency
        self._col_norms = np.linalg.norm(self.S, axis=0)

    def localize(
        self,
        residuals_per_sensor: Dict[str, float],
        top_k: int = 5,
    ) -> List[dict]:
        """
        Rank candidate leak nodes given a snapshot of pressure residuals.

        Parameters
        ----------
        residuals_per_sensor : dict {sensor_id: residual [m H₂O]}
            Typically the median residual over a detected leak window.
        top_k : int
            Number of top candidates to return. Default: 5.

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

        scores = []
        for j, node_id in enumerate(self.node_ids):
            s_j = self.S[:, j]  # sensitivity column for node j
            norm_j = self._col_norms[j]

            if norm_j < 1e-9:
                # Node has negligible pressure influence — not observable
                scores.append({
                    'node': node_id,
                    'estimated_leak_flow_ls': 0.0,
                    'score': 0.0,
                })
                continue

            # LS estimate of leak flow at node j
            q_hat = float(np.dot(s_j, r) / (norm_j ** 2))

            # Predicted residuals if leak is at j with flow q_hat
            r_pred = s_j * q_hat

            # Score: coefficient of determination (1 - normalized residual error)
            ss_res = np.sum((r - r_pred) ** 2)
            ss_tot = np.sum(r ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

            # Only positive leak flow is physically meaningful
            # (negative q_hat means node j cannot explain the observed drop)
            score = max(r2, 0.0) if q_hat > 0 else 0.0

            scores.append({
                'node': node_id,
                'estimated_leak_flow_ls': max(q_hat, 0.0),
                'score': float(score),
            })

        # Sort by score descending
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

        return self.localize(median_residuals, top_k=top_k)

    @staticmethod
    def localization_accuracy(
        candidates: List[dict],
        true_node: str,
        node_ids: List[str],
        adjacency: Optional[Dict[str, List[str]]] = None,
    ) -> dict:
        """
        Evaluate localization result against ground truth.

        Parameters
        ----------
        candidates : list of ranked candidate dicts
        true_node : str
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
        node_ranks = {c['node']: c['rank'] for c in candidates}
        rank_of_true = node_ranks.get(true_node, None)

        top1 = candidates[0]['node'] == true_node if candidates else False
        top3 = true_node in [c['node'] for c in candidates[:3]] if candidates else False

        hop_dist = None
        if adjacency is not None and candidates:
            best_guess = candidates[0]['node']
            hop_dist = _bfs_hop_distance(best_guess, true_node, adjacency)

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
