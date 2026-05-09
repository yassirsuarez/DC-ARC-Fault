#!/usr/bin/env python3
"""
rocket_validate_v3.py
"""
import pickle
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

INDICES = np.array(
    [list(c) for c in combinations(range(9), 3)],
    dtype=np.int32
)

def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_dataset(path, downsample=4):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    if downsample > 1:
        X = X[:, :, ::downsample]
    return X.astype(np.float32), y.astype(np.int64)

def rocket_transform_py(x_1d, tr):

    n_timepoints = len(x_1d)

    p0      = tr.parameter
    dil0    = np.array(p0[0], dtype=np.int32)
    n_fpd0  = np.array(p0[1], dtype=np.int32)
    bias0   = np.array(p0[2], dtype=np.float32)
    n_feat0 = int(84 * n_fpd0.sum())

    p1      = tr.parameter1
    dil1    = np.array(p1[0], dtype=np.int32)
    n_fpd1  = np.array(p1[1], dtype=np.int32)
    bias1   = np.array(p1[2], dtype=np.float32)
    n_feat1 = int(84 * n_fpd1.sum())

    n_features_per_kernel    = int(tr.n_features_per_kernel)
    total                    = (n_feat0 + n_feat1) * n_features_per_kernel
    n_features_per_transform = total // 2

    features = np.zeros(total, dtype=np.float32)

    def fill_set(x_in, dilations, n_fpd, biases, n_features, offset_base,
                 n_timepoints_orig=None):
        """
        n_timepoints_orig: lunghezza serie PRIMA del diff.
        Per set0: None -> usa len(x_in).
        Per set1: passa n_timepoints (250) perche' aeon usa
                  n_timepoints nei loop anche per la serie diff (249).
        """
        n_tp      = len(x_in)
        n_tp_loop = n_timepoints_orig if n_timepoints_orig is not None else n_tp

        feature_index_start = 0

        for d_idx in range(len(dilations)):
            _padding0 = d_idx % 2
            dilation  = int(dilations[d_idx])
            n_feat    = int(n_fpd[d_idx])
            padding   = ((9 - 1) * dilation) // 2

            A = -x_in
            G = x_in * 3.0

            C_alpha = np.zeros(n_tp, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, n_tp), dtype=np.float32)
            C_gamma[4] = G

            # 🔥 FIX: usa n_tp_loop (250) nei loop, non n_tp (249)
            start = dilation
            end   = n_tp_loop - padding

            for gi in range(4):
                e = min(end, n_tp)   # clip per non andare fuori bounds
                C_alpha[-e:] += A[:e]
                C_gamma[gi, -e:] = G[:e]
                end += dilation

            for gi in range(5, 9):
                s = min(start, n_tp)   # clip
                C_alpha[:-s] += A[s:]
                C_gamma[gi, :-s] = G[s:]
                start += dilation

            for k_idx in range(84):
                feature_index_end = feature_index_start + n_feat
                _padding1 = (_padding0 + k_idx) % 2

                i0, i1, i2 = INDICES[k_idx]
                C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]

                C_vec = C if _padding1 == 0 else C[padding:-padding]
                n_c   = len(C_vec)

                for feat_count in range(n_feat):
                    feature_index = feature_index_start + feat_count
                    bias = float(biases[feature_index])

                    ppv = last_val = 0
                    max_stretch = 0.0
                    mean_index = mean = 0.0

                    for j in range(n_c):
                        if C_vec[j] > bias:
                            ppv        += 1
                            mean_index += j
                            mean       += float(C_vec[j]) + bias
                        elif C_vec[j] < bias:
                            stretch = j - last_val
                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j

                    stretch = n_c - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    ppv_norm = float(ppv) / n_c if n_c > 0 else 0.0
                    mpv      = mean / ppv              if ppv > 0 else 0.0
                    mipv     = float(mean_index) / ppv if ppv > 0 else -1.0

                    fi = feature_index + offset_base
                    features[fi]                  = ppv_norm
                    features[fi + n_features]     = max_stretch
                    features[fi + 2 * n_features] = mpv
                    features[fi + 3 * n_features] = mipv

                feature_index_start = feature_index_end

    # Set 0: X raw
    fill_set(x_1d,
             dil0, n_fpd0, bias0,
             n_features=n_feat0,
             offset_base=0,
             n_timepoints_orig=None)       # usa len(x_1d) = 250

    # Set 1: diff(X,1) — passa n_timepoints originale (250)
    x_diff = np.diff(x_1d).astype(np.float32)
    fill_set(x_diff,
             dil1, n_fpd1, bias1,
             n_features=n_feat1,
             offset_base=n_features_per_transform,
             n_timepoints_orig=n_timepoints)   # 🔥 250, non 249

    return features


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("test")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--n-samples",  type=int, default=10)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    tr     = bundle["transformer"]
    X, y   = load_dataset(args.test, args.downsample)

    print("Transform nativo aeon...")
    F_native = tr.transform(X[:args.n_samples]).astype(np.float32)
    print(f"Native shape: {F_native.shape}")

    print("\nC-mirror Python...")
    errors = []

    print(f"\n{'idx':>4}  {'max_diff':>12}  {'mean_diff':>12}  {'match':>6}")
    print("-" * 50)

    for i in range(args.n_samples):
        x_1d  = X[i, 0, :]
        f_c   = rocket_transform_py(x_1d, tr)
        f_nat = F_native[i]
        n     = min(len(f_c), len(f_nat))
        diff  = np.abs(f_c[:n] - f_nat[:n])
        max_d  = diff.max()
        mean_d = diff.mean()
        match  = max_d < 1e-3
        if not match:
            errors.append(i)
        print(f"{i:>4}  {max_d:>12.6f}  {mean_d:>12.6f}  {'OK' if match else 'WARN':>6}")

    print("-" * 50)
    print(f"Match: {args.n_samples - len(errors)}/{args.n_samples}")

    if not errors:
        print("\nVALIDATION PASSED!")
        print("Il C-mirror e' corretto — pronti per generare rocket_transform.h finale")
    else:
        f_c   = rocket_transform_py(X[0, 0, :], tr)
        f_nat = F_native[0]
        diff  = np.abs(f_c - f_nat)
        worst = np.argsort(diff)[-10:]
        print("\nPeggiori 10 indici:")
        for idx in sorted(worst):
            print(f"  [{idx:6d}] c={f_c[idx]:.6f}  nat={f_nat[idx]:.6f}  diff={diff[idx]:.6f}")

        n_feat = 6216
        print("\nDiff per zona:")
        for nome, s, e in [
            ("PPV  set0", 0,          n_feat),
            ("LSPV set0", n_feat,     2*n_feat),
            ("MPV  set0", 2*n_feat,   3*n_feat),
            ("MIPV set0", 3*n_feat,   4*n_feat),
            ("PPV  set1", 4*n_feat,   5*n_feat),
            ("LSPV set1", 5*n_feat,   6*n_feat),
            ("MPV  set1", 6*n_feat,   7*n_feat),
            ("MIPV set1", 7*n_feat,   8*n_feat),
        ]:
            z = diff[s:e]
            print(f"  {nome}: max={z.max():.4f} mean={z.mean():.6f} n_errors={np.sum(z>1e-3)}")


if __name__ == "__main__":
    main()