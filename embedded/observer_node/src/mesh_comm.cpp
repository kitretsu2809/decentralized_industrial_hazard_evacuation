#include "mesh_comm.h"
#include <esp_now.h>
#include <WiFi.h>

// Broadcast address
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    // Optionally log status
}

void mesh_on_receive(const uint8_t* mac, const uint8_t* data, int len) {
    // Observer primarily sends, but can receive ACKs or configuration here
}

void mesh_init(const char* node_id) {
    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }

    esp_now_register_send_cb(OnDataSent);
    esp_now_register_recv_cb(mesh_on_receive);

    mesh_add_peer(broadcastAddress);
}

void mesh_add_peer(const uint8_t* mac) {
    esp_now_peer_info_t peerInfo;
    memcpy(peerInfo.peer_addr, mac, 6);
    peerInfo.channel = 1;  
    peerInfo.encrypt = false;
    
    if (esp_now_add_peer(&peerInfo) != ESP_OK){
        Serial.println("Failed to add peer");
        return;
    }
}

bool mesh_broadcast(const uint8_t* data, size_t len) {
    esp_err_t result = esp_now_send(broadcastAddress, data, len);
    return (result == ESP_OK);
}
