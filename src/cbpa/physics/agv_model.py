"""AGV (Automated Guided Vehicle) transfer and bottleneck model."""

from __future__ import annotations

from cbpa.config.defaults import (
    AGV_BASE_TRANSFER_S,
    AGV_CAPACITY_UPH,
    AGV_QUEUE_FACTOR,
)


class AGVModel:
    """Models inter-cell transfer time and AGV throughput capacity.

    The AGV shuttles finished products from Cell A's exit to Cell B's
    entry.  Transfer time increases with buffer occupancy (congestion).
    """

    def __init__(
        self,
        base_transfer_s: float = AGV_BASE_TRANSFER_S,
        queue_factor: float = AGV_QUEUE_FACTOR,
        capacity_uph: float = AGV_CAPACITY_UPH,
    ):
        self.base_transfer_s = base_transfer_s
        self.queue_factor = queue_factor
        self.capacity_uph = capacity_uph

    def transfer_time(self, buffer_occupancy: int = 0) -> float:
        """Compute AGV transfer time given current buffer occupancy.

        Args:
            buffer_occupancy: Number of units currently in the buffer.

        Returns:
            Transfer time in seconds.
        """
        return self.base_transfer_s + self.queue_factor * buffer_occupancy

    def utilization(
        self, cellA_throughput_uph: float, cellB_throughput_uph: float
    ) -> float:
        """Compute AGV utilization as fraction of capacity.

        The AGV must keep up with the slower of the two cells.

        Returns:
            Utilization fraction (0-1).  Values > 1.0 indicate bottleneck.
        """
        required_uph = min(cellA_throughput_uph, cellB_throughput_uph)
        if self.capacity_uph <= 0:
            return 1.0
        return round(required_uph / self.capacity_uph, 3)

    def is_bottleneck(
        self, cellA_throughput_uph: float, cellB_throughput_uph: float
    ) -> bool:
        """Check whether AGV is the factory throughput bottleneck."""
        return self.utilization(cellA_throughput_uph, cellB_throughput_uph) >= 1.0

    def effective_factory_throughput(
        self, cellA_throughput_uph: float, cellB_throughput_uph: float
    ) -> float:
        """Factory throughput limited by AGV capacity.

        Products flow A → AGV → B, so factory throughput is bounded by
        the minimum of all three capacities.
        """
        return min(cellA_throughput_uph, cellB_throughput_uph, self.capacity_uph)
