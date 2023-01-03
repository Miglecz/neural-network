package org.miglecz.ai.neuralnet;

import static java.lang.String.format;
import static java.util.logging.Level.CONFIG;
import java.util.Arrays;
import java.util.Optional;
import lombok.Builder;
import lombok.extern.flogger.Flogger;

@Flogger
public class FloatRecurringNetwork {
    private final int inputs;
    private final int outputs;
    private final float[] weights;
    private final float[] values; //values[0]=1 is bias
    private final int iterations;
    private final Activation<Float> activation;

    @Builder
    private FloatRecurringNetwork(
        Integer iterations,
        final Integer inputs,
        final Integer outputs,
        Integer nodes,
        final float[] weights,
        final Activation<Float> activation
    ) {
        iterations = Optional.ofNullable(iterations).orElse(1);
        nodes = Optional.ofNullable(nodes).orElseGet(() -> Math.max(inputs, outputs));
        if (iterations < 0) {
            throw new IllegalArgumentException("iterations should not be negative");
        }
        if (inputs > nodes) {
            throw new IllegalArgumentException("nodes should not be lower than inputs");
        }
        if (outputs > nodes) {
            throw new IllegalArgumentException("nodes should not be lower than outputs");
        }
        if (outputs <= 0) {
            throw new IllegalArgumentException();
        }
        final var weightsLength = ((nodes + 1) * nodes);
        if (weights != null && weights.length != weightsLength) {
            throw new IllegalArgumentException(format("weights length should be %d but was %d"
                , weightsLength
                , weights.length
            ));
        }
        this.iterations = iterations;
        this.inputs = inputs;
        this.outputs = outputs;
        values = new float[nodes + 1];
        values[0] = 1;
        this.activation = activation;
        this.weights = Optional.ofNullable(weights).orElse(new float[weightsLength]);
        log.at(CONFIG).log("construct: inputs=%d, outputs=%d, nodes=%d, activation=%s, iterations=%d, values=%s, weights=%s"
            , this.inputs
            , this.outputs
            , nodes
            , this.activation
            , this.iterations
            , Arrays.toString(values)
            , Arrays.toString(this.weights)
        );
    }

    public float[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public float[] getValues() {
        return Arrays.copyOf(values, values.length);
    }

    public FloatRecurringNetwork reset() {
        Arrays.fill(values, 1, values.length, 0);
        return this;
    }

    public float[] calculate(final float... inputs) {
        if (inputs.length != this.inputs) {
            throw new IllegalArgumentException("inputs.length should be " + this.inputs + " but was " + inputs.length);
        }
        copyInputs(inputs);
        for (int i = 0; i < iterations; ++i) {
            calculateNetwork();
        }
        final var outputs = getOutputs();
        log.at(CONFIG).log("calculate: %s -> %s"
            , Arrays.toString(inputs)
            , Arrays.toString(outputs)
        );
        return outputs;
    }

    @Override
    public String toString() {
        return format("%s(inputs=%d, outputs=%d, nodes=%d, activation=%s, iterations=%d, weights=%s)"
            , getClass().getSimpleName()
            , inputs
            , outputs
            , values.length - 1
            , activation
            , iterations
            , Arrays.toString(weights)
        );
    }

    private void calculateNetwork() {
        for (int i = 1; i < values.length; ++i) {
            values[i] = activation.apply(weightedSum(i));
        }
    }

    private float weightedSum(final int index) {
        var result = 0;
        final int n = values.length;
        final var offset = (index - 1) * n;
        for (int i = 0; i < n; ++i) {
            try {
                result += values[i] * weights[i + offset];
            } catch (final ArrayIndexOutOfBoundsException e) {
                log.at(CONFIG).log("index=%d, n=%d, offset=%d, len=%d-%d"
                    , index
                    , n
                    , offset
                    , weights.length
                    , i + offset
                );
                throw e;
            }
        }
        return result;
    }

    private void copyInputs(final float[] inputs) {
        System.arraycopy(inputs, 0, values, 1, inputs.length);
    }

    private float[] getOutputs() {
        return Arrays.copyOfRange(values, values.length - outputs, values.length);
    }
}
