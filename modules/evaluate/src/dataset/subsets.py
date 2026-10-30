"""Select named test subsets without changing development splits."""

from datasets import DatasetDict

from modules.evaluate.src.contracts.config import TestSubsets


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
