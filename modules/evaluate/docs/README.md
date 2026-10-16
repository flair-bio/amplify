# Evaluation Docs

Start with the [evaluation README](../README.md) for the
Prepare → Tune → Predict → Score pipeline, task types, controls, and config
reference. The guides below cover datasets and dataset-specific workflows.

| I want to... | Read |
|---|---|
| Run an evaluation or add a task type | [Evaluation README](../README.md) |
| Choose controls, or judge what a model learned from them | [Controls](controls.md) |
| Pick a dataset, or learn what its targets mean and which task uses it | [Dataset catalog](datasets.md) |
| Build or publish a dataset artifact, or check its provenance | [Dataset sourcing](dataset-sourcing.md) |
| Source or score ProteinGym assays | [ProteinGym](proteingym.md) |

Dataset overlays live in [`configs/datasets/`](../configs/datasets/); sourcing
configs live in [`configs/dataset_sourcing/`](../configs/dataset_sourcing/).

## Common questions

| Question | Answer |
|---|---|
| What do trunk, head, variant, and control mean? | [Glossary](../README.md#glossary) |
| Which datasets can I use with a given task? | [Datasets by task](datasets.md#datasets-by-task) |
| Which config overlays do I combine, and how do I run many models? | [Config files](../README.md#config-files), [Running many combinations](../README.md#running-many-combinations) |
| Where are my results, and how do I compare models? | [Output disk layout](../README.md#output-disk-layout), [Reading results](../README.md#reading-results) |
| Did the model learn anything beyond a baseline? | [Interpreting controls](controls.md#interpreting-results) |
| Which layer carries the signal? | [Layerwise linear probes](../README.md#layerwise-linear-probes) |
| Why did one run overwrite another? | [Output disk layout](../README.md#output-disk-layout): seeds and hyperparameters are not part of the run path |
| How do I add a dataset? | [Adding a new source](dataset-sourcing.md#adding-a-new-source), then a [dataset overlay](../README.md#new-dataset-task-or-model-overlay) |
| How do I add a task type? | [Adding a task type](../README.md#adding-a-task-type) |
| My run failed or hung | [Troubleshooting](../README.md#troubleshooting), [sourcing gotchas](dataset-sourcing.md#gotchas) |
