"""Select named test subsets without changing development splits."""

from datasets import DatasetDict

from modules.evaluate.src.contracts.config import (
    DatasetWorkspaceConfig,
    ExternalTestSetConfig,
    PrepareConfig,
    TestSubsets,
)
from modules.evaluate.src.dataset.dataloader import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.tasks import get_task


def resolve_test_subsets(
    workspace: DatasetWorkspaceConfig,
    prepare: PrepareConfig,
    subsets: TestSubsets,
    *,
    seed: int,
    dataset: DatasetDict | None = None,
    external: ExternalTestSetConfig | None = None,
) -> dict[str, DatasetDict]:
    """Resolve and validate a test source before applying task-aware selection."""
    task = get_task(workspace.task_type)
    if external is not None:
        if dataset is None:
            task.validate_test_source(
                external.dataset_repo_id, external.dataset_revision
            )
            dataset = resolve_external_data(
                external.dataset_repo_id,
                external.split,
                config_name=external.dataset_config_name,
                revision=external.dataset_revision,
            )
    else:
        if dataset is None:
            task.validate_test_source(
                workspace.dataset_repo_id or workspace.dataset_name,
                workspace.dataset_revision,
            )
        dataset = resolve_evaluation_data(
            workspace.dataset_repo_id or workspace.dataset_name,
            split=prepare.split,
            seed=seed,
            validation_fraction=prepare.validation_fraction,
            dataset=dataset,
            config_name=workspace.dataset_config_name,
            revision=workspace.dataset_revision,
        )
    task.validate_test_dataset(dataset)
    return select_test_subsets(dataset, subsets, group_column=task.test_group_column)


def select_test_subsets(
    dataset: DatasetDict,
    subsets: TestSubsets,
    *,
    group_column: str | None = None,
) -> dict[str, DatasetDict]:
    test = dataset["test"]
    selected = {}
    for name, filters in subsets.items():
        missing = set(filters) - set(test.column_names)
        if missing:
            raise ValueError(
                f"Test subset '{name}': missing columns {sorted(missing)}."
            )
        indices = [
            index
            for index, row in enumerate(test.select_columns(list(filters)))
            if all(
                row[column] == value
                and (not isinstance(value, bool) or isinstance(row[column], bool))
                for column, value in filters.items()
            )
        ]
        if not indices:
            raise ValueError(f"Test subset '{name}' is empty.")
        if group_column is not None:
            groups = test[group_column]
            selected_groups = {groups[index] for index in indices}
            if len(indices) != sum(group in selected_groups for group in groups):
                raise ValueError(
                    f"Test subset '{name}' must retain all scoring units of each chain."
                )
        selected[name] = DatasetDict({**dataset, "test": test.select(indices)})
    return selected
