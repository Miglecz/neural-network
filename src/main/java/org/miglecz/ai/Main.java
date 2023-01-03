package org.miglecz.ai;

import java.util.List;

class Main {
    public static void main(final String[] args) {
        final var data = List.of(
            //not
            //TrainingData.builder().input(new float[]{0}).output(new float[]{1}).build()
            //, TrainingData.builder().input(new float[]{1}).output(new float[]{0}).build()
            //or
            //TrainingData.builder().input(new float[]{0, 0}).output(new float[]{0}).build()
            //, TrainingData.builder().input(new float[]{0, 1}).output(new float[]{1}).build()
            //, TrainingData.builder().input(new float[]{1, 0}).output(new float[]{1}).build()
            //, TrainingData.builder().input(new float[]{1, 1}).output(new float[]{1}).build()
            //and
            //TrainingData.builder().input(new float[]{0, 0}).output(new float[]{0}).build()
            //, TrainingData.builder().input(new float[]{0, 1}).output(new float[]{0}).build()
            //, TrainingData.builder().input(new float[]{1, 0}).output(new float[]{0}).build()
            //, TrainingData.builder().input(new float[]{1, 1}).output(new float[]{1}).build()
            //xor
            TrainingData.builder().input(new float[]{0, 0}).output(new float[]{0}).build()
            , TrainingData.builder().input(new float[]{0, 1}).output(new float[]{1}).build()
            , TrainingData.builder().input(new float[]{1, 0}).output(new float[]{1}).build()
            , TrainingData.builder().input(new float[]{1, 1}).output(new float[]{0}).build()
        );
        System.out.printf("data=%s%n", data);
        final var solution = new Trainer(data).train();
        System.out.printf("best=%s%n", solution);
    }
}
