# Evaluation Docs

Start with the [evaluation README](../README.md) for the
prepare, fit, predict, score, and save workflow, task types, controls, and config
reference. The guides below cover datasets and dataset-specific workflows.

| I want to... | Read |
|---|---|
| Run an evaluation or add a task type | [Evaluation README](../README.md) |
| Choose controls, or judge what a model learned from them | [Controls](controls.md) |
| Pick a dataset, or learn what its targets mean and which task uses it | [Dataset catalog](datasets.md) |
| Build or publish a dataset artifact | [Downstream data preparation](https://github.com/flair-bio/amplify-downstream-data-prep/tree/main) (separate repository) |
| Run or understand ProteinGym evaluation | [ProteinGym](proteingym.md) |

Dataset overlays live in [`configs/datasets/`](../configs/datasets/). This
module loads prepared artifacts; dataset sourcing, standardization, and
publication live in the separate data-preparation repository. Keep dataset
cards and pinned artifact revisions with your experiment for provenance.

## Common questions

| Question | Answer |
|---|---|
| What do trunk, head, variant, and control mean? | [Glossary](../README.md#glossary) |
| Which datasets can I use with a given task? | [Datasets by task](datasets.md#datasets-by-task) |
| Which config overlays do I combine, and how do I run many models? | [Config files](../README.md#config-files), [Running many combinations](../README.md#running-many-combinations) |
| How do I score a fitted model on another test set? | [External test sets](../README.md#external-test-sets) |
| Where are my results, and how do I compare models? | [Output disk layout](../README.md#output-disk-layout), [Reading results](../README.md#reading-results) |
| Did the model learn anything beyond a baseline? | [Interpreting controls](controls.md#interpreting-results) |
| Which layer carries the signal? | [Layerwise linear probes](../README.md#layerwise-linear-probes) |
| Why did one run overwrite another? | [Output disk layout](../README.md#output-disk-layout): seeds and variants are isolated; replacing an existing run requires `--overwrite` |
| How do I add a dataset? | Prepare the artifact in the [data-preparation repository](https://github.com/flair-bio/amplify-downstream-data-prep/tree/main), then add a [dataset overlay](../README.md#new-dataset-task-or-model-overlay) |
| How do I add a task type? | [Adding a task type](../README.md#adding-a-task-type) |
| My run failed or hung | [Troubleshooting](../README.md#troubleshooting) |
