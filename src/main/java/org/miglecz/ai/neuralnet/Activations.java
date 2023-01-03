package org.miglecz.ai.neuralnet;

import java.util.function.UnaryOperator;
import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;

@UtilityClass
public class Activations {
    public static Activation<Float> TANH = namedOperator("Tanh", x -> (float) Math.tanh(x));
    public static Activation<Float> RELU = namedOperator("ReLU", x -> Math.max(0, x));

    @SneakyThrows
    private <T> Activation<T> namedOperator(final String name, final UnaryOperator<T> op) {
        return new Activation<>(new UnaryOperator<>() {
            @Override
            public T apply(final T x) {
                return op.apply(x);
            }

            @Override
            public String toString() {
                return name;
            }
        });
    }
}
