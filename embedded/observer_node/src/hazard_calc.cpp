#include "hazard_calc.h"
#include <math.h>

float normalize_hazard(float raw, float min_val, float max_val) {
    if (raw < min_val) return min_val;
    if (raw > max_val) return max_val;
    return raw;
}

float compute_fire_hazard(float temp, float smoke, float co) {
    // Temperature contribution: sigmoid centered around 60°C
    float temp_h = 1.0f / (1.0f + expf(-0.1f * (temp - 60.0f)));
    
    // Smoke contribution: linear
    float smoke_h = normalize_hazard(smoke, 0.0f, 1.0f);
    
    // CO contribution
    float co_h = 0.0f;
    if (co <= 50.0f) {
        co_h = (co / 50.0f) * 0.5f;
    } else {
        co_h = 0.5f + ((co - 50.0f) / 150.0f) * 0.5f;
    }
    co_h = normalize_hazard(co_h, 0.0f, 1.0f);

    float max_h = temp_h;
    if (smoke_h > max_h) max_h = smoke_h;
    if (co_h > max_h) max_h = co_h;
    
    float mean_h = (temp_h + smoke_h + co_h) / 3.0f;
    
    return normalize_hazard(max_h * 0.6f + mean_h * 0.4f, 0.0f, 1.0f);
}

float combine_hazards(float* hazards, int count) {
    if (count <= 0) return 0.0f;
    
    float max_h = hazards[0];
    float sum_h = hazards[0];
    
    for (int i = 1; i < count; i++) {
        if (hazards[i] > max_h) {
            max_h = hazards[i];
        }
        sum_h += hazards[i];
    }
    
    float mean_h = sum_h / (float)count;
    return normalize_hazard(max_h * 0.7f + mean_h * 0.3f, 0.0f, 1.0f);
}

bool is_impassable(float hazard) {
    // Uses constant from config logically, but defaults to 0.95
    return hazard >= 0.95f;
}
