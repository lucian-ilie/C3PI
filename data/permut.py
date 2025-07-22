import numpy as np
from itertools import permutations
from random import shuffle

def hamming_distance(a, b):
    return np.sum(a != b)

def far_permutations(base, n):
    perms = [base.copy()]
    used = set()
    used.add(tuple(base))

    while len(perms) < n:
        best_candidate = None
        best_min_distance = -1

        candidates = [np.random.permutation(base) for _ in range(5000)]

        for candidate in candidates:
            t = tuple(candidate)
            if t in used:
                continue
            min_dist = min(hamming_distance(candidate, p) for p in perms)
            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_candidate = candidate

        if best_candidate is not None:
            perms.append(best_candidate)
            used.add(tuple(best_candidate))
        else:
            print("Could not find a better permutation.")
            break

    return np.array(perms)

# Base row
base = np.arange(0, 16)

# Generate 100 diverse permutations
array_100x16 = far_permutations(base, 100)

# Save to .npy file
np.save("permutations_16.npy", array_100x16)
