/**
 * Actuator Node - ESP32-S3 Firmware
 * 
 * Receives commands from Router Node via mesh
 * Drives LED matrix (WS2812B) for directional arrows
 * Controls relay for magnetic door lock
 * Plays WAV alerts via I2S speaker
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <FastLED.h>

#define NODE_ID "actuator_f1_door_a"
#define PIN_LED_DATA 18
#define PIN_RELAY 23
#define NUM_LEDS 64

CRGB leds[NUM_LEDS];

enum SystemState {
    NORMAL,
    EVACUATION,
    LOCKDOWN
};

SystemState currentState = NORMAL;

// Structure to receive data
typedef struct struct_message {
    char target_id[32];
    int command; // 0: Normal, 1: Evacuate, 2: Lockdown
    int direction; // For LEDs
    int door_lock; // 0: Unlock, 1: Lock
} struct_message;

struct_message incomingData;

void onDataRecv(const uint8_t * mac, const uint8_t *incomingDataBuf, int len) {
    memcpy(&incomingData, incomingDataBuf, sizeof(incomingData));
    
    // Only process if it's meant for us (or broadcast)
    if (strcmp(incomingData.target_id, NODE_ID) == 0 || strcmp(incomingData.target_id, "BROADCAST") == 0) {
        
        // Update State
        if (incomingData.command == 0) currentState = NORMAL;
        else if (incomingData.command == 1) currentState = EVACUATION;
        else if (incomingData.command == 2) currentState = LOCKDOWN;
        
        // Update Door
        digitalWrite(PIN_RELAY, incomingData.door_lock ? HIGH : LOW);
        
        // Update LEDs
        fill_solid(leds, NUM_LEDS, CRGB::Black); // clear
        if (currentState == EVACUATION) {
            fill_solid(leds, NUM_LEDS, CRGB::Green); // Mock arrow
        } else if (currentState == LOCKDOWN) {
            fill_solid(leds, NUM_LEDS, CRGB::Red); // Mock stop
        }
        FastLED.show();
        
        // I2S Audio alert placeholder
        if (currentState == EVACUATION) {
            Serial.println("Playing Evacuation Audio...");
        } else if (currentState == LOCKDOWN) {
            Serial.println("Playing Lockdown Audio...");
        }
    }
}

void setup() {
    Serial.begin(115200);
    
    pinMode(PIN_RELAY, OUTPUT);
    digitalWrite(PIN_RELAY, LOW); // Default unlocked
    
    FastLED.addLeds<WS2812B, PIN_LED_DATA, GRB>(leds, NUM_LEDS);
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    FastLED.show();
    
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    
    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW Init Failed");
        return;
    }
    
    esp_now_register_recv_cb(onDataRecv);
    Serial.println("Actuator Node Ready");
}

void loop() {
    // Main logic driven by callbacks
    delay(100);
}
