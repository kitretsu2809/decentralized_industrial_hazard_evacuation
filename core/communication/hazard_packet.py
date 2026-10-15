"""Four-byte hazard gossip packet used between adjacent nodes.

Bit layout, least-significant bit first::

    0..7    eight-direction safe-corridor mask (N, NE, E, SE, S, SW, W, NW)
    8..15   fused hazard score, quantized from [0, 1]
    16..23  crowd density, quantized from [0, 2]
    24..27  hazard type code
    28..29  protocol version
    30..31  time-to-live hop count
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

from core.graph.directions import DIRECTION_NAMES


PACKET_SIZE_BYTES = 4
PROTOCOL_VERSION = 1

HAZARD_TYPE_CODES: Dict[str, int] = {
    "NONE": 0,
    "FIRE": 1,
    "THERMAL_FIRE": 1,
    "SMOKE": 2,
    "GAS": 3,
    "GAS_RELEASE": 3,
    "TOXIC_PLUME": 3,
    "CHEMICAL_SPILL": 4,
    "STRUCTURAL_COLLAPSE": 5,
    "COLLAPSE": 5,
    "EXPLOSION": 6,
    "WEAPON": 7,
    "UNKNOWN": 15,
}
HAZARD_CODE_NAMES = {
    0: "NONE", 1: "FIRE", 2: "SMOKE", 3: "GAS_RELEASE",
    4: "CHEMICAL_SPILL", 5: "STRUCTURAL_COLLAPSE",
    6: "EXPLOSION", 7: "WEAPON", 15: "UNKNOWN",
}


def _quantize(value: float, maximum: float) -> int:
    bounded = max(0.0, min(maximum, float(value)))
    return int(round((bounded / maximum) * 255.0))


@dataclass(frozen=True)
class HazardPacket32:
    direction_mask: int
    hazard: float
    crowd_density: float
    hazard_type: str = "NONE"
    version: int = PROTOCOL_VERSION
    ttl: int = 2

    def to_bytes(self) -> bytes:
        return pack_hazard_packet(self)

    def public_dict(self) -> dict:
        raw = self.to_bytes()
        active = [
            name for index, name in enumerate(DIRECTION_NAMES)
            if self.direction_mask & (1 << index)
        ]
        return {
            "size_bits": 32,
            "hex": raw.hex().upper(),
            "bits": format(int.from_bytes(raw, "big"), "032b"),
            "direction_mask": format(self.direction_mask, "08b"),
            "active_directions": active,
            "hazard": round(self.hazard, 4),
            "crowd_density": round(self.crowd_density, 4),
            "hazard_type": self.hazard_type,
            "version": self.version,
            "ttl": self.ttl,
        }


def pack_hazard_packet(packet: HazardPacket32) -> bytes:
    """Encode a packet into exactly four network-order bytes."""
    if not 0 <= int(packet.direction_mask) <= 0xFF:
        raise ValueError("direction_mask must fit in eight bits")
    if not 0 <= int(packet.version) <= 0x03:
        raise ValueError("version must fit in two bits")
    if not 0 <= int(packet.ttl) <= 0x03:
        raise ValueError("ttl must fit in two bits")
    type_code = HAZARD_TYPE_CODES.get(str(packet.hazard_type).upper(), 15)
    value = (
        int(packet.direction_mask)
        | (_quantize(packet.hazard, 1.0) << 8)
        | (_quantize(packet.crowd_density, 2.0) << 16)
        | ((type_code & 0x0F) << 24)
        | ((int(packet.version) & 0x03) << 28)
        | ((int(packet.ttl) & 0x03) << 30)
    )
    return value.to_bytes(PACKET_SIZE_BYTES, byteorder="big", signed=False)


def unpack_hazard_packet(raw: Union[bytes, bytearray]) -> HazardPacket32:
    """Decode and validate a four-byte network-order packet."""
    if len(raw) != PACKET_SIZE_BYTES:
        raise ValueError(f"Hazard packet must be exactly {PACKET_SIZE_BYTES} bytes")
    value = int.from_bytes(raw, byteorder="big", signed=False)
    direction_mask = value & 0xFF
    hazard_q = (value >> 8) & 0xFF
    crowd_q = (value >> 16) & 0xFF
    type_code = (value >> 24) & 0x0F
    version = (value >> 28) & 0x03
    ttl = (value >> 30) & 0x03
    return HazardPacket32(
        direction_mask=direction_mask,
        hazard=hazard_q / 255.0,
        crowd_density=(crowd_q / 255.0) * 2.0,
        hazard_type=HAZARD_CODE_NAMES.get(type_code, "UNKNOWN"),
        version=version,
        ttl=ttl,
    )
