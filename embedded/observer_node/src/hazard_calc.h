#ifndef HAZARD_CALC_H
#define HAZARD_CALC_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

float normalize_hazard(float raw, float min_val, float max_val);
float compute_fire_hazard(float temp, float smoke, float co);
float combine_hazards(float* hazards, int count);
bool is_impassable(float hazard);

#ifdef __cplusplus
}
#endif

#endif
