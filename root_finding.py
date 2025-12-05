import os
import numpy as np
import cupy as cp
from math import ceil
from tqdm import trange
import tqdm


pairwise_kernel_block_code = r'''
extern "C" __global__
void pairwise_torus_sqdist_block(
    const float* __restrict__ pos,
    float* __restrict__ dists,
    int n_rows, int n_cols,
    int d, int row_start, int col_start)
{
    int li = blockDim.y * blockIdx.y + threadIdx.y;
    int lj = blockDim.x * blockIdx.x + threadIdx.x;

    if (li >= n_rows || lj >= n_cols) return;

    int i = row_start + li;
    int j = col_start + lj;

    // Invalid: cannot attach to j >= i, so make it huge
    if (j >= i) {
        dists[li * n_cols + lj] = 1e30f;
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < d; k++) {
        float diff = fabsf(pos[i*d + k] - pos[j*d + k]);
        diff = fminf(diff, 1.0f - diff);
        acc += diff * diff;
    }
    dists[li * n_cols + lj] = acc;
}
''';

_pairwise_kernel_block = cp.RawKernel(
    pairwise_kernel_block_code,
    "pairwise_torus_sqdist_block"
)


def nearest_attachment_tree_blockwise_gpu(
        n: int,
        d: int = 2,
        block_cols: int = 4096,
        mem_budget_gb: float = 4.0,
        seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Node positions (unit torus)
    positions = np.random.rand(n, d).astype(np.float32)
    pos_gpu = cp.asarray(positions)

    # Parents array
    parents = np.empty(n, dtype=np.int64)
    parents[0] = -1  # root

    BYTES_PER_FLOAT = 4
    mem_bytes = int(mem_budget_gb * (1024**3))

    # Compute max number of rows per distance block
    max_batch_rows = max(1, mem_bytes // (block_cols * BYTES_PER_FLOAT))
    max_batch_rows = min(max_batch_rows, 256)

    tx, ty = 16, 16

    # Loop over row batches
    for row_start in range(1, n, max_batch_rows):
        batch_rows = min(max_batch_rows, n - row_start)

        block_mins = []
        block_indices = []

        for col_start in range(0, n, block_cols):
            col_block = min(block_cols, n - col_start)

            dmat = cp.empty((batch_rows, col_block), dtype=cp.float32)

            bx = ceil(col_block / tx)
            by = ceil(batch_rows / ty)

            # Launch GPU kernel
            _pairwise_kernel_block(
                (bx, by, 1),
                (tx, ty, 1),
                (
                    pos_gpu,
                    dmat,
                    np.int32(batch_rows),
                    np.int32(col_block),
                    np.int32(d),
                    np.int32(row_start),
                    np.int32(col_start)
                )
            )

            # Block-level argmin
            block_min_vals = cp.min(dmat, axis=1)
            block_min_idx_local = cp.argmin(dmat, axis=1)
            block_min_idx_global = block_min_idx_local + col_start

            block_mins.append(block_min_vals)
            block_indices.append(block_min_idx_global)

        # Combine all blocks
        mins_mat = cp.stack(block_mins, axis=1)
        idx_mat = cp.stack(block_indices, axis=1)

        best_block = cp.argmin(mins_mat, axis=1)
        best_idx = idx_mat[cp.arange(batch_rows), best_block]

        parents[row_start:row_start+batch_rows] = cp.asnumpy(best_idx)

    return parents, positions

def attachment_tree_alpha_beta_blockwise_gpu(
        n: int,
        d: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
        block_cols: int = 4096,
        mem_budget_gb: float = 4.0,
        seed=None):
    """
    GPU blockwise tree generation with attachment probability proportional to
    (degree)^alpha * (1/distance)^beta, using torus distances.
    """

    if seed is not None:
        np.random.seed(seed)

    # Node positions in unit torus
    positions = np.random.rand(n, d).astype(np.float32)
    pos_gpu = cp.asarray(positions)

    parents = np.empty(n, dtype=np.int64)
    parents[0] = -1  # root
    degree = np.zeros(n, dtype=np.int64)
    degree[0] = 1

    BYTES_PER_FLOAT = 4
    mem_bytes = int(mem_budget_gb * (1024**3))
    max_batch_rows = max(1, mem_bytes // (block_cols * BYTES_PER_FLOAT))
    max_batch_rows = min(max_batch_rows, 256)

    for row_start in range(1, n, max_batch_rows):
        batch_rows = min(max_batch_rows, n - row_start)
        parent_choices = np.empty(batch_rows, dtype=np.int64)

        # Compute distances to previous nodes in column blocks
        for col_start in range(0, row_start, block_cols):
            col_block = min(block_cols, row_start - col_start)
            # block distance: shape (batch_rows, col_block, d)
            diff = pos_gpu[row_start:row_start+batch_rows, None, :] - pos_gpu[col_start:col_start+col_block][None, :, :]
            diff = cp.minimum(cp.abs(diff), 1.0 - cp.abs(diff))  # torus
            dists = cp.sqrt(cp.sum(diff**2, axis=2))  # shape (batch_rows, col_block)

            # Compute weights
            deg_block = cp.asarray(degree[col_start:col_start+col_block], dtype=cp.float32)
            weights_block = (deg_block[None, :]**alpha) * (dists**(-beta))

            if col_start == 0:
                weights_accum = weights_block
            else:
                weights_accum = cp.concatenate([weights_accum, weights_block], axis=1)

        # Normalize weights along rows
        weights_accum /= cp.sum(weights_accum, axis=1, keepdims=True)

        # Sample parents for batch
        cum_weights = cp.cumsum(weights_accum, axis=1)
        rand_vals = cp.asarray(np.random.rand(batch_rows), dtype=cp.float32)[:, None]
        parent_choices_gpu = cp.argmax(cum_weights >= rand_vals, axis=1)
        parent_choices[:] = cp.asnumpy(parent_choices_gpu)

        # Assign parents and update degrees
        parents[row_start:row_start+batch_rows] = parent_choices
        np.add.at(degree, parent_choices, 1)
        degree[row_start:row_start+batch_rows] = 1

        # free GPU memory for next batch
        del weights_accum, cum_weights, diff, dists

    return parents, positions

def attachment_tree_alpha_beta_gpu(
        n: int,
        d: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
        block_cols: int = 4096,
        mem_budget_gb: float = 4.0,
        seed=None):
    """
    Generate a tree where attachment probability ~ (degree)^alpha * (1/distance)^beta.
    
    Returns:
        parents: np.ndarray of shape (n,)
        positions: np.ndarray of shape (n, d)
    """

    if seed is not None:
        np.random.seed(seed)

    positions = np.random.rand(n, d).astype(np.float32)
    pos_gpu = cp.asarray(positions)

    parents = np.empty(n, dtype=np.int64)
    parents[0] = -1  # root
    degree = np.zeros(n, dtype=np.int64)
    degree[0] = 1

    BYTES_PER_FLOAT = 4
    mem_bytes = int(mem_budget_gb * (1024**3))
    max_batch_rows = max(1, mem_bytes // (block_cols * BYTES_PER_FLOAT))
    max_batch_rows = min(max_batch_rows, 256)

    tx, ty = 16, 16

    # Precompute pairwise distance on GPU for small batches
    for i in trange(1, n, desc="Building tree"):
        # Distances from node i to all previous nodes
        diff = cp.abs(pos_gpu[i] - pos_gpu[:i])
        diff = cp.minimum(diff, 1.0 - diff)  # torus
        dists = cp.sqrt(cp.sum(diff**2, axis=1))

        # Compute weights: (degree)^alpha * (1/dist)^beta
        deg_cp = cp.asarray(degree[:i], dtype=cp.float32)
        weights = (deg_cp ** alpha) * (dists ** (-beta))
        weights = cp.asnumpy(weights)
        weights /= np.sum(weights)

        # Sample parent according to weights
        parent = np.random.choice(i, p=weights)
        parents[i] = parent
        degree[parent] += 1
        degree[i] = 1

    return parents, positions


def compute_psi_values_scorer(parents):
    n = len(parents)
    children = [[] for _ in range(n)]
    for u, p in enumerate(parents):
        if p >= 0:
            children[p].append(u)

    subtree = np.ones(n, dtype=int)

    # DFS
    stack = [0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)

    # Bottom-up DP
    for u in reversed(order):
        for v in children[u]:
            subtree[u] += subtree[v]

    psi = np.zeros(n, dtype=int)
    for u in range(n):
        C = [subtree[v] for v in children[u]]
        if u != 0:
            C.append(n - subtree[u])
        psi[u] = max(C) if C else 0
    return psi

def largest_root_child_fraction(parents):
    """
    Given a parent array, return the size of the largest child subtree
    of the root (node 0) divided by n.
    """
    n = len(parents)

    # Build children list
    children = [[] for _ in range(n)]
    for u, p in enumerate(parents):
        if p >= 0:
            children[p].append(u)

    # Compute subtree sizes
    subtree = np.ones(n, dtype=int)
    stack = [0]
    order = []

    # DFS
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)

    # Bottom-up subtree accumulation
    for u in reversed(order):
        for v in children[u]:
            subtree[u] += subtree[v]

    # Largest subtree among root's children
    root_children = children[0]
    if not root_children:
        return 0.0

    max_sub = max(subtree[v] for v in root_children)
    return max_sub / n


# ---------- helper: torus squared distance ----------
def torus_dist_sq_batch(pts_a, pts_b):
    """
    Compute squared torus distances between pts_a (shape (A,2)) and pts_b (shape (B,2))
    Returns array shape (A,B) of squared distances.
    Use with care: memory = A*B floats.
    """
    # pts_a[:,None,:] - pts_b[None,:,:] => shape (A,B,2)
    diff = np.abs(pts_a[:, None, :] - pts_b[None, :, :])
    diff = np.minimum(diff, 1.0 - diff)
    return np.sum(diff * diff, axis=2)


# ---------- function: largest root-child indices set ----------
def largest_root_child_indices(parents):
    """
    Given parents array for a prefix of size m, return a boolean mask
    of length m indicating which nodes belong to the largest root-child subtree.
    (node 0 is the root)
    """
    m = len(parents)
    children = [[] for _ in range(m)]
    for u, p in enumerate(parents):
        if p >= 0 and p < m:
            children[p].append(u)

    # compute subtree sizes bottom-up
    subtree = np.ones(m, dtype=int)
    stack = [0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)
    for u in reversed(order):
        for v in children[u]:
            subtree[u] += subtree[v]

    root_children = children[0]
    if not root_children:
        return np.zeros(m, dtype=bool)

    # find largest child index
    sizes = [subtree[v] for v in root_children]
    idx_max_child = root_children[int(np.argmax(sizes))]
    # collect all nodes in that child's subtree
    mask = np.zeros(m, dtype=bool)
    stack = [idx_max_child]
    while stack:
        u = stack.pop()
        mask[u] = True
        for v in children[u]:
            stack.append(v)
    return mask


import cupy as cp

def torus_dist_sq_gpu(points, seeds):
    """
    points: (B,2) CuPy array
    seeds:  (M,2) CuPy array
    Returns: (B,M) squared torus distances (GPU)
    """
    # points[:,None,:] - seeds[None,:,:] → shape (B,M,2)
    diff = cp.abs(points[:, None, :] - seeds[None, :, :])
    diff = cp.minimum(diff, 1.0 - diff)
    return cp.sum(diff * diff, axis=2)


def estimate_basin_area_gpu(seed_positions,
                            subset_mask,
                            samples=20000,
                            batch_size=2048,
                            rng=None):
    """
    GPU Monte-Carlo estimation of the Voronoi basin area of a subset of seeds.

    seed_positions: (m,2) NumPy or CuPy array
    subset_mask: boolean mask of length m
    samples: total Monte Carlo samples
    batch_size: sample batch size (controls GPU memory)
    rng_seed: optional seed for reproducibility

    Returns: float (estimated fraction of torus assigned to subset)
    """
    if isinstance(seed_positions, np.ndarray):
        seeds_gpu = cp.asarray(seed_positions, dtype=cp.float32)
    else:
        seeds_gpu = seed_positions.astype(cp.float32)

    subset_mask_gpu = cp.asarray(subset_mask, dtype=cp.bool_)
    m = seeds_gpu.shape[0]

    if rng is not None:
        cp.random.seed(int(0))

    samples = int(samples)
    batch_size = int(batch_size)

    total_hits = 0
    total_count = 0

    n_batches = (samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        # draw uniform points on torus
        b = min(batch_size, samples - total_count)
        pts = cp.random.random((b, 2), dtype=cp.float32)

        # compute distances batchwise
        d2 = torus_dist_sq_gpu(pts, seeds_gpu)

        nearest = cp.argmin(d2, axis=1)
        hits = cp.count_nonzero(subset_mask_gpu[nearest])

        total_hits += hits.get()       # bring only the integer to CPU
        total_count += b

    return total_hits / total_count

def largest_subtree_basin_fraction_per_file(save_dir, tree_sizes, mc_samples=20000, mc_batch=2000):
    import numpy as np
    from tqdm import tqdm
    rng = np.random.default_rng(12345)
    results = {}

    for n in tree_sizes:
        file_path = os.path.join(save_dir, f"torus_trees_n={n}.npz")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping n={n}")
            continue

        # Load compressed .npz
        data = np.load(file_path, allow_pickle=True)
        parents_arr = data['parents']
        points_arr = data['points']

        max_fraction = -1
        best_trial = None

        # Find trial with largest root-child fraction
        for t in tqdm(range(len(parents_arr)), desc=f"Finding largest child n={n}"):
            parents = parents_arr[t]
            frac = largest_root_child_fraction(parents)
            if frac > max_fraction:
                max_fraction = frac
                best_trial = t

        # Compute Voronoi fraction for largest subtree of best trial
        parents_best = parents_arr[best_trial]
        points_best = points_arr[best_trial]
        mask = largest_root_child_indices(parents_best)

        voronoi_fraction = estimate_basin_area_gpu(
            points_best, mask,
            samples=mc_samples,
            batch_size=mc_batch,
            rng=rng
        )

        results[n] = {
            "best_trial": best_trial,
            "largest_child_fraction": max_fraction,
            "voronoi_fraction": voronoi_fraction
        }

        # Free memory
        del parents_arr, points_arr

    return results

def continue_tree(parents_existing, positions_existing, n_new, keep_n=None, block_cols=4096, mem_budget_gb=4.0, seed=None):
    """
    Continue building a tree from an existing set of nodes using nearest_attachment_tree_blockwise_gpu,
    optionally keeping only the first `keep_n` nodes.

    Args:
        parents_existing: np.ndarray of shape (n0,) with parent indices of the existing tree
        positions_existing: np.ndarray of shape (n0, d) with positions of existing nodes
        n_new: total size of tree after continuation (n_new > keep_n)
        keep_n: number of nodes from the original tree to keep (default: keep all)
        block_cols: GPU block column size
        mem_budget_gb: memory budget for GPU blocks
        seed: optional random seed

    Returns:
        parents_new: np.ndarray of shape (n_new,)
        positions_new: np.ndarray of shape (n_new, d)
    """
    n0 = len(parents_existing)
    
    # Determine how many nodes to keep
    if keep_n is None:
        keep_n = n0
    if keep_n > n0 or keep_n < 1:
        raise ValueError("keep_n must be between 1 and the size of the existing tree")
    if n_new <= keep_n:
        raise ValueError("n_new must be greater than keep_n")

    d = positions_existing.shape[1]
    np.random.seed(seed)
    
    # Slice original arrays to keep only the first keep_n nodes
    parents_kept = parents_existing[:keep_n].copy()
    positions_kept = positions_existing[:keep_n].copy()
    
    # Initialize new positions array
    positions_new = np.empty((n_new, d), dtype=np.float32)
    positions_new[:keep_n] = positions_kept
    positions_new[keep_n:] = np.random.rand(n_new - keep_n, d).astype(np.float32)
    
    # Initialize new parents array
    parents_new = np.empty(n_new, dtype=np.int64)
    parents_new[:keep_n] = parents_kept
    
    # Transfer positions to GPU
    pos_gpu = cp.asarray(positions_new)
    
    BYTES_PER_FLOAT = 4
    mem_bytes = int(mem_budget_gb * (1024**3))
    max_batch_rows = max(1, mem_bytes // (block_cols * BYTES_PER_FLOAT))
    max_batch_rows = min(max_batch_rows, 256)
    
    tx, ty = 16, 16
    
    # Build tree for new nodes in batches
    for row_start in range(keep_n, n_new, max_batch_rows):
        batch_rows = min(max_batch_rows, n_new - row_start)
        
        block_mins = []
        block_indices = []
        
        for col_start in range(0, row_start, block_cols):
            col_block = min(block_cols, row_start - col_start)
            
            dmat = cp.empty((batch_rows, col_block), dtype=cp.float32)
            
            bx = ceil(col_block / tx)
            by = ceil(batch_rows / ty)
            
            _pairwise_kernel_block(
                (bx, by, 1),
                (tx, ty, 1),
                (
                    pos_gpu,
                    dmat,
                    np.int32(batch_rows),
                    np.int32(col_block),
                    np.int32(d),
                    np.int32(row_start),
                    np.int32(col_start)
                )
            )
            
            block_min_vals = cp.min(dmat, axis=1)
            block_min_idx_local = cp.argmin(dmat, axis=1)
            block_min_idx_global = block_min_idx_local + col_start
            
            block_mins.append(block_min_vals)
            block_indices.append(block_min_idx_global)
        
        # Combine block results
        mins_mat = cp.stack(block_mins, axis=1)
        idx_mat = cp.stack(block_indices, axis=1)
        best_block = cp.argmin(mins_mat, axis=1)
        best_idx = idx_mat[cp.arange(batch_rows), best_block]
        
        parents_new[row_start:row_start+batch_rows] = cp.asnumpy(best_idx)
    
    return parents_new, positions_new
