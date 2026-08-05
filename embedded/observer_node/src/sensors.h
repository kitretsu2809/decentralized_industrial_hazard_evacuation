#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>
#include "config.h"

void sensors_init();
float read_gas_ppm();
float read_temperature();
float read_humidity();
int read_presence();

#endif
