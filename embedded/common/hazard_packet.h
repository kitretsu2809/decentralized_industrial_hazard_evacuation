#pragma once

#include <Arduino.h>

// Wire order matches core/communication/hazard_packet.py exactly:
// byte 0: TTL[7:6], version[5:4], hazard type[3:0]
// byte 1: crowd density quantized from 0..2
// byte 2: hazard score quantized from 0..1
// byte 3: safe directions N,NE,E,SE,S,SW,W,NW as bits 0..7
struct __attribute__((packed)) HazardPacket32 {
    uint8_t protocol;
    uint8_t crowd;
    uint8_t hazard;
    uint8_t direction_mask;
};

static_assert(sizeof(HazardPacket32) == 4, "HazardPacket32 must remain four bytes");

inline uint8_t quantize_unit(float value, float maximum) {
    const float bounded = constrain(value, 0.0f, maximum);
    return static_cast<uint8_t>(roundf((bounded / maximum) * 255.0f));
}

inline HazardPacket32 make_hazard_packet(
    uint8_t direction_mask,
    float hazard,
    float crowd_density,
    uint8_t hazard_type = 0,
    uint8_t ttl = 2
) {
    HazardPacket32 packet{};
    packet.protocol = static_cast<uint8_t>(
        ((ttl & 0x03u) << 6) | (1u << 4) | (hazard_type & 0x0Fu)
    );
    packet.crowd = quantize_unit(crowd_density, 2.0f);
    packet.hazard = quantize_unit(hazard, 1.0f);
    packet.direction_mask = direction_mask;
    return packet;
}
