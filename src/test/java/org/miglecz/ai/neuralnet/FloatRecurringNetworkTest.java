package org.miglecz.ai.neuralnet;

import static com.google.common.truth.Truth.assertThat;
import static org.miglecz.ai.neuralnet.Activations.RELU;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class FloatRecurringNetworkTest {
    private static final float[] XOR_WEIGHTS = new float[]{0.47649288f, 2.5589762f, 2.062996f, -5.11413f, 9.448477f, -9.811835f, 1.7756319f, 2.1085844f, 2.5155783f, -0.3580103f, -6.535014f, 4.312193f};

    @Test(dataProvider = "data")
    void calculateShouldReturnExpectedOutputs(final int nodes, final Activation<Float> activation, final float[] inputs, final float[] outputs, final float[] weights) {
        // Given
        final var network = FloatRecurringNetwork.builder()
            .inputs(inputs.length)
            .outputs(outputs.length)
            .nodes(nodes)
            .activation(activation)
            .weights(weights)
            .build();
        // When
        final var result = network.calculate(inputs);
        // Then
        assertThat(result).isEqualTo(outputs);
    }

    @DataProvider
    private Object[][] data() {
        return new Object[][]{
            new Object[]{3, RELU, new float[]{0, 0}, new float[]{0}, XOR_WEIGHTS}
            , new Object[]{3, RELU, new float[]{0, 1}, new float[]{1}, XOR_WEIGHTS}
            , new Object[]{3, RELU, new float[]{1, 0}, new float[]{1}, XOR_WEIGHTS}
            , new Object[]{3, RELU, new float[]{1, 1}, new float[]{0}, XOR_WEIGHTS}
        };
    }
}
