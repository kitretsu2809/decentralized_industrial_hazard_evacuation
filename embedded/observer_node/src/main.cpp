/**
 * Observer Node - ESP32-S3 Firmware
 * 
 * Reads environmental sensors (MQ-2 gas, DHT22 temp/humidity, IR ToF presence)
 * and broadcasts hazard data over ESP-NOW mesh to Router Nodes.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include "sensors.h"
#include "hazard_calc.h"
#include "mesh_comm.h"
#include "config.h"
#include "hazard_packet.h"

// State variables
unsigned long lastSensorReadTime = 0;
unsigned long lastBroadcastTime = 0;
unsigned long lastDebugPrintTime = 0;

float current_gas = 0.0f;
float current_temp = 0.0f;
float current_hum = 0.0f;
int current_presence = 0;
float current_hazard = 0.0f;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("Starting LBP Observer Node...");

    // Initialize WiFi in Station mode for ESP-NOW
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Initialize subsystems
    sensors_init();
    mesh_init(NODE_ID);
    
    Serial.println("Initialization complete.");
}

void loop() {
    unsigned long currentMillis = millis();

    // 1. Read Sensors (100Hz max, per config)
    if (currentMillis - lastSensorReadTime >= SENSOR_READ_INTERVAL_MS) {
        lastSensorReadTime = currentMillis;
        
        current_gas = read_gas_ppm();
        current_temp = read_temperature();
        current_hum = read_humidity();
        current_presence = read_presence();
        
        // 2. Compute hazard score via hazard_calc
        // co_ppm approximation for mq2 gas
        float fire_hazard = compute_fire_hazard(current_temp, current_gas / 1000.0f, current_gas);
        
        // Combine with other potential hazards if any
        float hazards[] = {fire_hazard};
        current_hazard = combine_hazards(hazards, 1);
    }

    // 3 & 4. Package data and broadcast via ESP-NOW
    if (currentMillis - lastBroadcastTime >= BROADCAST_INTERVAL_MS) {
        lastBroadcastTime = currentMillis;
        
        const float crowd_density = constrain(
            static_cast<float>(current_presence) / static_cast<float>(NODE_CAPACITY),
            0.0f,
            2.0f
        );
        const uint8_t hazard_type = current_temp >= TEMP_ALARM_C
            ? 1  // FIRE
            : (current_gas >= GAS_ALARM_PPM ? 3 : 0);  // GAS_RELEASE or NONE
        HazardPacket32 packet = make_hazard_packet(
            NODE_DIRECTION_MASK,
            current_hazard,
            crowd_density,
            hazard_type
        );

        mesh_broadcast(reinterpret_cast<uint8_t*>(&packet), sizeof(packet));
    }

    // 5. Print debug info to serial
    if (currentMillis - lastDebugPrintTime >= DEBUG_PRINT_INTERVAL_MS) {
        lastDebugPrintTime = currentMillis;
        Serial.printf("[%s] Hazard: %.3f | Temp: %.1fC | Gas: %.1f ppm | Presence: %d\n", 
            NODE_ID, current_hazard, current_temp, current_gas, current_presence);
    }
}
