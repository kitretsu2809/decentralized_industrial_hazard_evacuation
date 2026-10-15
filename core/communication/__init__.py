"""Compact communication protocols shared by simulator and edge nodes."""

from .hazard_packet import HazardPacket32, pack_hazard_packet, unpack_hazard_packet

__all__ = ["HazardPacket32", "pack_hazard_packet", "unpack_hazard_packet"]
