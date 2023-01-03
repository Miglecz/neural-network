package org.miglecz.ai;

import lombok.Builder;
import lombok.Value;

@Value
@Builder
class TrainingData {
    float[] input;
    float[] output;
}
