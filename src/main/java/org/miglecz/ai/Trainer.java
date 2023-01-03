package org.miglecz.ai;

import static java.util.logging.Level.CONFIG;
import static org.miglecz.ai.neuralnet.Activations.RELU;
import static org.miglecz.optimization.genetic.operator.Crossovers.uniformCrossover;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import lombok.extern.flogger.Flogger;
import org.miglecz.ai.neuralnet.FloatRecurringNetwork;
import org.miglecz.optimization.Solution;
import org.miglecz.optimization.genetic.GeneticOptimizationBuilder;
import org.miglecz.optimization.stream.Collectors;
import org.miglecz.optimization.stream.TakeWhiles;

@Flogger
class Trainer {
    private final int nodes = 3;
    private final Random random = new Random();
    private final List<TrainingData> data;

    public Trainer(final List<TrainingData> data) {
        this.data = data;
    }

    public Solution<FloatRecurringNetwork> train() {
        return GeneticOptimizationBuilder.builder(FloatRecurringNetwork.class)
            .withPopulation(50)
            .withElite(1)
            .withImmigrant(50)
            .withFactory(this::factory)
            .withMutant(50, this::mutate)
            .withOffspring(50, this::crossover)
            .withFitness(this::fitness)
            .withRandom(random)
            .build()
            .stream()
            //.peek(System.out::println)
            .takeWhile(TakeWhiles.belowScore(0))
            .takeWhile(TakeWhiles.progressingIteration(10_000))
            .collect(Collectors.toBestSolution())
            ;
    }

    public double fitness(final FloatRecurringNetwork network) {
        final var result = 0 - data.stream()
            .mapToDouble(d -> {
                network.reset();
                return diffSquareSum(d.getOutput(), network.calculate(d.getInput()));
            })
            .sum();
        log.at(CONFIG).log("fitness: %s", result);
        return result;
    }

    private FloatRecurringNetwork factory() {
        final var inputs = data.get(0).getInput().length;
        final var outputs = data.get(0).getOutput().length;
        final var nodes = Math.max(this.nodes, Math.max(inputs, outputs));
        final var weightsLength = (nodes + 1) * nodes;
        final var weights = new float[weightsLength];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = randomWeight();
        }
        return baseBuilder()
            .weights(weights)
            .build();
    }

    private FloatRecurringNetwork mutate(final FloatRecurringNetwork network) {
        final float[] weights = network.getWeights();
        final var index = random.nextInt(weights.length);
        weights[index] = randomWeight();
        log.at(CONFIG).log("mutate: %s -> %s"
            , Arrays.toString(network.getWeights())
            , Arrays.toString(weights)
        );
        return baseBuilder().weights(weights).build();
    }

    private FloatRecurringNetwork crossover(final FloatRecurringNetwork parent1, final FloatRecurringNetwork parent2) {
        final var w1 = parent1.getWeights();
        final var w2 = parent2.getWeights();
        final var result = baseBuilder()
            .weights(uniformCrossover(random, w1, w2))
            .build();
        log.at(CONFIG).log("crossover: %s X %s -> %s"
            , Arrays.toString(w1)
            , Arrays.toString(w2)
            , Arrays.toString(result.getWeights())
        );
        return result;
    }

    private float randomWeight() {
        return random.nextFloat(-10, 10);
    }

    private static double diffSquareSum(final float[] actual, final float[] expected) {
        return IntStream.range(0, actual.length)
            .mapToDouble(i -> actual[i] - expected[i])
            .map(d -> d * d)
            .sum();
    }

    private FloatRecurringNetwork.FloatRecurringNetworkBuilder baseBuilder() {
        return FloatRecurringNetwork.builder()
            .inputs(data.get(0).getInput().length)
            .outputs(data.get(0).getOutput().length)
            //.iterations(2)
            .nodes(nodes)
            .activation(RELU);
    }
}
