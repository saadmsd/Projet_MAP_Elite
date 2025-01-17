"""This file contains util functions and a class to define
a repertoire, used to store individuals in the MAP-Elites
algorithm as well as several variants."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpy.random import RandomState
from sklearn.cluster import KMeans

from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

from flax import struct

@struct.dataclass
class UCBRepertoire(MapElitesRepertoire):
    total_counts: int = struct.field(pytree_node=False, default=0)
    rewards: jnp.ndarray = struct.field(pytree_node=True, default=None)
    counts: jnp.ndarray = struct.field(pytree_node=True, default=None)
    survivals: jnp.ndarray = struct.field(pytree_node=True, default=None)

    def __init__(self, genotypes: jnp.ndarray, fitnesses: jnp.ndarray, descriptors: jnp.ndarray, centroids: jnp.ndarray):
        print("UCBBBBBBBBBBBBBB")
        super().__init__(genotypes, fitnesses, descriptors, centroids)
        object.__setattr__(self, 'total_counts', 0)
        object.__setattr__(self, 'rewards', jnp.zeros(fitnesses.shape))
        object.__setattr__(self, 'counts', jnp.zeros(fitnesses.shape))
        object.__setattr__(self, 'survivals', jnp.zeros(fitnesses.shape))

    def sample_by_indices(self, indices: List[int]) -> Genotype:
        """
        Sample genotypes from the repertoire using specified indices.

        Params:
            indices: List of indices to sample.

        Returns:
            A batch of genotypes sampled from the repertoire.
        """
        indices = jnp.array(indices)  # Convert list of indices to JAX array
        return jax.tree_util.tree_map(lambda x: x[indices], self.genotypes)
        
    def update_ucb(self, indices: List[int], rewards: jnp.ndarray, survivals: jnp.ndarray):
        self.total_counts += len(indices)
        self.rewards = self.rewards.at[indices].add(rewards)
        self.counts = self.counts.at[indices].add(1)
        self.survivals = self.survivals.at[indices].add(survivals)

    def ucb_selection(self, c=1.0):
        ucb_values = self.rewards / (self.counts + 1e-6) + c * jnp.sqrt(jnp.log(self.total_counts) / (self.counts + 1e-6))
        return jnp.argmax(ucb_values)

    def get_cell_info(self, descriptor: Descriptor) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get the genotypes, fitnesses, and descriptors of individuals in the specified cell.

        Params:
            descriptor: The descriptor defining the cell.

        Returns:
            A tuple of (genotypes, fitnesses, descriptors) of individuals in the cell.
        """
        cell_indices = jnp.where(jnp.all(self.descriptors == descriptor, axis=1))[0]
        cell_genotypes = self.genotypes[cell_indices]
        cell_fitnesses = self.fitnesses[cell_indices]
        cell_descriptors = self.descriptors[cell_indices]
        return cell_genotypes, cell_fitnesses, cell_descriptors