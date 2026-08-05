#include "sensors.h"
#include <DHT.h>

DHT dht(PIN_DHT22, DHT22);

void sensors_init() {
    pinMode(PIN_MQ2_ANALOG, INPUT);
    pinMode(PIN_IR_TOF, INPUT);
    dht.begin();
}

float read_gas_ppm() {
    // Simple mock calibration for MQ-2
    int raw = analogRead(PIN_MQ2_ANALOG);
    // Convert 12-bit ADC to approximated PPM
    return (float)raw * (1000.0f / 4095.0f);
}

float read_temperature() {
    float t = dht.readTemperature();
    if (isnan(t)) return 25.0f; // default safe fallback
    return t;
}

float read_humidity() {
    float h = dht.readHumidity();
    if (isnan(h)) return 50.0f;
    return h;
}

int read_presence() {
    // IR ToF mock reading (digital for simple presence)
    return digitalRead(PIN_IR_TOF) == HIGH ? 1 : 0;
}
