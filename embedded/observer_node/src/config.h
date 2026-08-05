#ifndef CONFIG_H
#define CONFIG_H

// Node Identity
#define NODE_ID "observer_f1_office1"
#define NODE_FLOOR 1

// Sensor Pins
#define PIN_MQ2_ANALOG 34
#define PIN_DHT22 4
#define PIN_IR_TOF 5

// Timing
#define SENSOR_READ_INTERVAL_MS 10   // 100 Hz
#define BROADCAST_INTERVAL_MS 100     // 10 Hz
#define DEBUG_PRINT_INTERVAL_MS 1000  // 1 Hz

// Hazard Thresholds
#define GAS_ALARM_PPM 50.0f
#define TEMP_ALARM_C 60.0f
#define IMPASSABLE_THRESHOLD 0.95f

// Mesh
#define MESH_CHANNEL 1

#endif
