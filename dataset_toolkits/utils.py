from typing import *
import hashlib
import numpy as np


def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]



# reference of object categories
cat_ref = {
    "Table": 1,
    "Dishwasher": 2,
    "StorageFurniture": 3,
    "Refrigerator": 4,
    "WashingMachine": 5,
    "Microwave": 6,
    "Oven": 7,
    "Safe": 8,
}

# reference of semantic labels for each part
sem_ref = {
    "fwd": {
        "door": 1,
        "drawer": 2,
        "base": 3,
        "handle": 4,
        "wheel": 5,
        "knob": 6,
        "shelf": 7,
        "tray": 8
    },
    "bwd": {
        1: "door",
        2: "drawer",
        3: "base",
        4: "handle",
        5: "wheel",
        6: "knob",
        7: "shelf",
        8: "tray"
    }
}

# reference of joint types for each part
joint_ref = {
    "fwd": {
       "fixed": 1,
        "revolute": 2,
        "prismatic": 3,
        "screw": 4,
        "continuous": 5 
    },
    "bwd": {
        1: "fixed",
        2: "revolute",
        3: "prismatic",
        4: "screw",
        5: "continuous"
    } 
}


import plotly.express as px
# pallette for joint type color
joint_color_ref = px.colors.qualitative.Set1
# pallette for graph node color
graph_color_ref = px.colors.qualitative.Bold + px.colors.qualitative.Prism
# pallette for semantic label color
semantic_color_ref = px.colors.qualitative.Vivid_r