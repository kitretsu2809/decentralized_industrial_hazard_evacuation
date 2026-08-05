#ifndef MESH_COMM_H
#define MESH_COMM_H

#include <Arduino.h>

void mesh_init(const char* node_id);
void mesh_add_peer(const uint8_t* mac);
bool mesh_broadcast(const uint8_t* data, size_t len);
void mesh_on_receive(const uint8_t* mac, const uint8_t* data, int len);

#endif
