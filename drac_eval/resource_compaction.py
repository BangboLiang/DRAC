"""Schedule-wide stable channel compaction and physical binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sparse_realization import DirectionRequest, RealizationResult


@dataclass(frozen=True)
class ChannelBinding:
    segment: int
    src: int
    tx_channel: int
    dst: int
    rx_channel: int


@dataclass(frozen=True)
class CompactionResult:
    reserved_tx: np.ndarray
    reserved_rx: np.ndarray
    exposed_tx: np.ndarray
    exposed_rx: np.ndarray
    reserved_bundles: np.ndarray
    exposed_bundles: np.ndarray
    bindings: tuple[tuple[ChannelBinding, ...], ...]
    request_map: tuple[DirectionRequest, ...]

    @property
    def total_stable_directional_pool(self) -> int:
        return int(self.reserved_tx.sum() + self.reserved_rx.sum())


def compact_schedule(
    configurations: Sequence[np.ndarray],
    inventory_tx: np.ndarray,
    inventory_rx: np.ndarray,
    realization_results: Sequence[RealizationResult] | None = None,
) -> CompactionResult:
    tx_inventory = np.asarray(inventory_tx, dtype=int)
    rx_inventory = np.asarray(inventory_rx, dtype=int)
    if tx_inventory.shape != rx_inventory.shape or np.any(tx_inventory < 0) or np.any(rx_inventory < 0):
        raise ValueError("invalid physical channel inventory")
    size = len(tx_inventory)
    if not configurations:
        zero = np.zeros(size, dtype=int)
        return CompactionResult(zero, zero, tx_inventory.copy(), rx_inventory.copy(), zero, np.minimum(tx_inventory, rx_inventory), (), ())
    for configuration in configurations:
        if configuration.shape != (size, size) or np.any(configuration < 0):
            raise ValueError("configuration shape or value is invalid")
    reserved_tx = np.max(np.stack([matrix.sum(axis=1) for matrix in configurations]), axis=0).astype(int)
    reserved_rx = np.max(np.stack([matrix.sum(axis=0) for matrix in configurations]), axis=0).astype(int)
    if np.any(reserved_tx > tx_inventory) or np.any(reserved_rx > rx_inventory):
        raise ValueError("integer schedule cannot fit the physical inventory")
    reserved_bundles = np.maximum(reserved_tx, reserved_rx)
    bundle_inventory = np.minimum(tx_inventory, rx_inventory)
    if np.any(reserved_bundles > bundle_inventory):
        raise ValueError("integer schedule cannot fit bidirectional bundle inventory")

    all_bindings: list[tuple[ChannelBinding, ...]] = []
    for segment, configuration in enumerate(configurations):
        next_tx = np.zeros(size, dtype=int)
        next_rx = np.zeros(size, dtype=int)
        bindings: list[ChannelBinding] = []
        for src in range(size):
            for dst in range(size):
                for _ in range(int(configuration[src, dst])):
                    tx_channel = int(next_tx[src])
                    rx_channel = int(next_rx[dst])
                    if tx_channel >= reserved_tx[src] or rx_channel >= reserved_rx[dst]:
                        raise AssertionError("stable pool is insufficient for a segment")
                    bindings.append(ChannelBinding(segment, src, tx_channel, dst, rx_channel))
                    next_tx[src] += 1
                    next_rx[dst] += 1
        all_bindings.append(tuple(bindings))

    requests: dict[tuple[int, int], DirectionRequest] = {}
    for result in realization_results or ():
        for request in result.requests:
            key = (request.src, request.dst)
            previous = requests.get(key)
            if previous is None:
                requests[key] = request
            else:
                requests[key] = DirectionRequest(
                    request.src,
                    request.dst,
                    max(previous.marginal_gain, request.marginal_gain),
                    previous.aggregate_demand + request.aggregate_demand,
                )
    request_map = tuple(sorted(requests.values(), key=lambda item: (-item.marginal_gain, item.src, item.dst)))
    return CompactionResult(
        reserved_tx,
        reserved_rx,
        tx_inventory - reserved_tx,
        rx_inventory - reserved_rx,
        reserved_bundles,
        bundle_inventory - reserved_bundles,
        tuple(all_bindings),
        request_map,
    )


def verify_compaction_sufficiency(
    result: CompactionResult, configurations: Sequence[np.ndarray]
) -> bool:
    for configuration in configurations:
        if np.any(configuration.sum(axis=1) > result.reserved_tx):
            return False
        if np.any(configuration.sum(axis=0) > result.reserved_rx):
            return False
    return True
