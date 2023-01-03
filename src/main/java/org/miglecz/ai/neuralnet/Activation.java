package org.miglecz.ai.neuralnet;

import java.util.function.UnaryOperator;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Delegate;

@RequiredArgsConstructor
public class Activation<T> implements UnaryOperator<T> {
    @Delegate
    private final UnaryOperator<T> op;

    @Override
    public String toString() {
        return op.toString();
    }

    @Override
    public boolean equals(final Object obj) {
        return op.equals(obj);
    }

    @Override
    public int hashCode() {
        return op.hashCode();
    }
}
