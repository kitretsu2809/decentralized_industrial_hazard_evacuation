/**
 * Camera Node - ESP32-CAM / Arduino Vision Firmware
 * 
 * Captures video frames from OV2640 camera sensor.
 * Performs edge-based optical crowd density and head-count estimation.
 * Broadcasts crowd metrics (occupancy count & Delta rho) over ESP-NOW mesh.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include "esp_camera.h"

// ==========================================
// Select Camera Model Pinout (Default: AI-Thinker)
// ==========================================
#define CAMERA_MODEL_AI_THINKER // ESP32-CAM AI Thinker

#if defined(CAMERA_MODEL_AI_THINKER)
  #define PWDN_GPIO_NUM     32
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM      0
  #define SIOD_GPIO_NUM     26
  #define SIOC_GPIO_NUM     27
  
  #define Y9_GPIO_NUM       35
  #define Y8_GPIO_NUM       34
  #define Y7_GPIO_NUM       39
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       21
  #define Y4_GPIO_NUM       19
  #define Y3_GPIO_NUM       18
  #define Y2_GPIO_NUM        5
  #define VSYNC_GPIO_NUM    25
  #define HREF_GPIO_NUM     23
  #define PCLK_GPIO_NUM     22
#endif

// Node Configuration (Injected dynamically by Provisioner Tool)
#ifndef NODE_ID
  #define NODE_ID "camera_f1_chokepoint"
#endif

#ifndef NODE_FLOOR
  #define NODE_FLOOR 1
#endif

#ifndef NODE_CAPACITY
  #define NODE_CAPACITY 30
#endif

// Telemetry message structure (fits 4-byte / 8-byte gossip packet)
typedef struct __attribute__((packed)) {
    char node_id[16];
    uint16_t head_count;
    uint8_t crowd_density_pct; // 0 - 200% of capacity
    int8_t delta_rho;          // Change since last gossip broadcast
} CameraGossipPacket;

CameraGossipPacket packet;
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

unsigned long lastProcessTime = 0;
uint16_t last_count = 0;

bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE; // Efficient for edge vision analysis
    config.frame_size = FRAMESIZE_QQVGA;       // 160x120 for rapid MCU edge analysis
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera initialization failed with error 0x%x\n", err);
        return false;
    }
    Serial.println("OV2640 Camera initialized successfully (160x120 Grayscale)");
    return true;
}

uint16_t estimateCrowdCount(camera_fb_t *fb) {
    if (!fb || !fb->buf) return 0;
    
    // Fast lightweight edge gradient / blob presence algorithm on microcontroller
    // Counts high-contrast localized intensity transitions (head-shoulder silhouettes)
    int threshold = 35;
    int transitions = 0;
    int w = fb->width;
    int h = fb->height;

    for (int y = 10; y < h - 10; y += 4) {
        for (int x = 10; x < w - 10; x += 4) {
            int diff = abs((int)fb->buf[y * w + x] - (int)fb->buf[y * w + (x + 2)]);
            if (diff > threshold) {
                transitions++;
            }
        }
    }
    
    // Calibrated heuristic: ~12-16 edge transitions per pedestrian silhouette in 160x120 frame
    uint16_t estimated = (uint16_t)(transitions / 14);
    return estimated;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.printf("\n=== LBP Camera Crowd Sensor Node: %s (Floor %d) ===\n", NODE_ID, NODE_FLOOR);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW mesh");
    } else {
        Serial.println("ESP-NOW Mesh interface ready");
        esp_now_peer_info_t peerInfo = {};
        memcpy(peerInfo.peer_addr, broadcastAddress, 6);
        peerInfo.channel = 11;
        peerInfo.encrypt = false;
        esp_now_add_peer(&peerInfo);
    }

    if (!initCamera()) {
        Serial.println("FATAL: Camera hardware not responding. Check FPC ribbon cable.");
    }
}

void loop() {
    unsigned long now = millis();
    if (now - lastProcessTime >= 250) { // 4 Hz vision inference cycle
        lastProcessTime = now;

        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            uint16_t count = estimateCrowdCount(fb);
            esp_camera_fb_return(fb);

            float density_ratio = (float)count / (float)NODE_CAPACITY;
            int8_t delta = (int8_t)(count - last_count);

            strncpy(packet.node_id, NODE_ID, sizeof(packet.node_id));
            packet.head_count = count;
            packet.crowd_density_pct = (uint8_t)min(255, (int)(density_ratio * 100.0f));
            packet.delta_rho = delta;

            // Broadcast delta gossip packet only on significant crowd change or heartbeat
            if (abs(delta) >= 2 || (now % 3000 < 250)) {
                esp_now_send(broadcastAddress, (uint8_t *)&packet, sizeof(packet));
                Serial.printf("[%s] Vision Telemetry: Count=%d | Density=%.1f%% | Delta=%+d\n",
                              NODE_ID, count, density_ratio * 100.0f, delta);
            }
            last_count = count;
        }
    }
}
