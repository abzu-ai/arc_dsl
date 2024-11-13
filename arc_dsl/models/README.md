# Models

This directory contains the deep-learning models used.

## Overview

The models are based on the transformer architecture, implemented in Keras with a TensorFlow backend.

The models have encoders specifically designed to handle ARC grids, and decoders that generate DSL sentences. This makes us able to get high performance models with a small number of parameters.

In our previous experiments with LLM's and multimodal LLM's, we found that these small models capture grid semantics better than the LLM's. This is due to `PositionalGridEmbedding` that encodes both row and column position of element in the grid.

The reduced output language in form of DSL sentences makes it easier train and achieve meaningful output. The DSL is both simple and expressive, and makes it trivial to generate infinite training data in form of DSL accompanied with ARC grids. Therefore we are not limited to the few training examples provided by the ARC dataset.

There are two working models and one that is in progress.

### Model: Decompostion

An auto-regressive transformer model that decomposes an ARC grid into bitmap objects.

- **Input (Encoder)**: A normalized ARC grid.
- **Input (Decoder)**: A start token.
- **Output**: DSL sentences representing the decomposition, such as:
```
(MAKE-BITMAP 1 1 2 2 BLUE 13)(MAKE-BITMAP 4 4 2 2 GREEN 15)
```

#### Training Data:

An infinite generator produces ARC grids and their corresponding decompositions as DSL sentences.

### Model: Ideas

_Incomplete!_

### Model: Pattern completion

Another auto-regressive transformer model, that given parameterized decomposition and an ARC grid, infers the actual decomposition that matches the grid.

- **Input (Encoder)**: A normalized ARC grid.
- **Input (Decoder)**: A parameterized DSL sentence with placeholders (? and ¿) for unknown values, such as:
```
<START>(MAKE-BITMAP 1 ? 2 2 ? 13)(MAKE-BITMAP ? ? 2 2 GREEN ¿)<COMPLETION>
```
- **Output**: A DSL sentence where the placeholders have been filled out, such as:
```
<START>(MAKE-BITMAP 1 ? 2 2 ? 13)(MAKE-BITMAP ? ? 2 2 GREEN ¿)<COMPLETION>(MAKE-BITMAP 1 1 2 2 BLUE 13)(MAKE-BITMAP 4 4 2 2 GREEN 15)<END>
```

#### Training Data:

Again an infinite generator produces ARC grids along with their parameterized decompositions and the corresponding actual decompositions as completions.

## Normalized ARC grids

We normalize ARC grids by converting each grid into a 30×30 sequence of numbers between 0 and 10:

- Each consecutive chunk of 30 numbers represents a row in the grid.
- Grids smaller than 30x30 are padded with zeros on each row.
- 0 represents padding
- 1 to 10 represent the ARC colors, offset by 1.

### Encoder embeddings

The `PositionalGridEmbedding`-layer has 3 embedding layers. On for the sequence itself, then additonal embedding layers that encodes row position and column position for each token in this sequence.

## Training data

We generate infite random sequences of training examples as `(gridlisp, rendered grid)` pairs, and we do not touch the actual training data available in the ARC challenge.

## Training

A training session is started by running either `train_decomposition_model.py` or `train_pattern_completion_model.py` in the root of this repository. The models will be saved in the folder `checkpoints`.

Models were trained on a single NVIDIA H100 SMX5 GPU with 80GB of memory, and took a few hours to achieve good performance.

Trained models can be found in the folder `arc_dsl/trained_models`.

### Parameters

The decomposition model have the following parameters:
```text/plain
 Total params: 2,816,492 (10.74 MB)
 Trainable params: 2,816,492 (10.74 MB)
 Non-trainable params: 0 (0.00 B)
```

The pattern completion model have the following parameters:
```text/plain
 Total params: 3,760,044 (14.34 MB)
 Trainable params: 3,760,044 (14.34 MB)
 Non-trainable params: 0 (0.00 B)
```
