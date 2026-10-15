import unittest

from core.communication.hazard_packet import HazardPacket32, unpack_hazard_packet
from core.graph.directions import DIRECTION_NAMES, assign_direction_slots, connected_direction_mask
from embedded.router_node.src.mesh_client import MeshClient


class HazardPacketTests(unittest.TestCase):
    def test_packet_is_exactly_four_bytes_and_round_trips(self):
        packet = HazardPacket32(
            direction_mask=0b10100101,
            hazard=0.72,
            crowd_density=1.25,
            hazard_type="FIRE",
            version=1,
            ttl=2,
        )
        raw = packet.to_bytes()
        self.assertEqual(len(raw), 4)
        decoded = unpack_hazard_packet(raw)
        self.assertEqual(decoded.direction_mask, 0b10100101)
        self.assertEqual(decoded.hazard_type, "FIRE")
        self.assertAlmostEqual(decoded.hazard, 0.72, delta=1 / 255)
        self.assertAlmostEqual(decoded.crowd_density, 1.25, delta=2 / 255)
        self.assertEqual(decoded.version, 1)
        self.assertEqual(decoded.ttl, 2)

    def test_wire_byte_order_matches_embedded_header(self):
        packet = HazardPacket32(
            direction_mask=0xA5,
            hazard=1.0,
            crowd_density=2.0,
            hazard_type="GAS_RELEASE",
            version=1,
            ttl=2,
        )
        self.assertEqual(packet.to_bytes().hex().upper(), "93FFFFA5")

    def test_eight_neighbors_receive_unique_direction_bits(self):
        positions = {
            "c": (0, 0), "n": (0, -1), "ne": (1, -1), "e": (1, 0),
            "se": (1, 1), "s": (0, 1), "sw": (-1, 1), "w": (-1, 0),
            "nw": (-1, -1),
        }
        slots = assign_direction_slots("c", positions.keys() - {"c"}, positions)
        self.assertEqual(tuple(slots), ("n", "ne", "e", "se", "s", "sw", "w", "nw"))
        self.assertEqual(connected_direction_mask(slots), 0xFF)
        self.assertEqual(len(DIRECTION_NAMES), 8)

    def test_more_than_eight_neighbors_is_rejected(self):
        positions = {"c": (0, 0), **{f"n{i}": (i + 1, 0) for i in range(9)}}
        with self.assertRaises(ValueError):
            assign_direction_slots("c", [f"n{i}" for i in range(9)], positions)

    def test_embedded_mesh_client_uses_the_same_four_byte_codec(self):
        client = MeshClient("router_a")
        packet = HazardPacket32(
            direction_mask=0b10100101,
            hazard=0.42,
            crowd_density=1.25,
            hazard_type="GAS_RELEASE",
        )

        raw = client.broadcast(packet)
        decoded = client.ingest("router_b", raw)

        self.assertEqual(len(raw), 4)
        self.assertEqual(decoded["direction_mask"], "10100101")
        self.assertEqual(client.get_neighbor_states()["router_b"]["size_bits"], 32)


if __name__ == "__main__":
    unittest.main()
