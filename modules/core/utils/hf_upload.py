import logging
from pathlib import Path
from typing import Literal
from huggingface_hub import HfApi, create_repo


logger = logging.getLogger(__name__)


def upload_to_hf(
    folder_path: str | Path,
    repo_id: str,
    repo_type: Literal["model", "dataset", "space"] = "model",
    path_in_repo: str | None = None,
    token: str | None = None,
    revision: str | None = None,
    commit_message: str = "Upload folder",
    private: bool = True,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> None:
    """
    Uploads a local folder or file (dataset/model artifact) to the Hugging Face Hub.

    Args:
        folder_path (str | Path): The local path to a folder or a single file to upload.
        repo_id (str): The destination repo on HF (e.g., "username/my-awesome-model").
        repo_type (str): The type of repository. Must be 'model', 'dataset', or 'space'.
        path_in_repo (str, optional): Destination directory in the target repo.
        token (str, optional): Your HF User Access Token. If None, uses the cached CLI token.
        revision (str, optional): The target Git branch, tag, or commit on the Hub.
                                  If None, defaults to the 'main' branch.
        commit_message (str): The Git commit message.
        private (bool): Whether the repository should be private if newly created.
        allow_patterns (list, optional): List of glob patterns to include (e.g., ["*.json", "*.safetensors"]).
        ignore_patterns (list, optional): List of glob patterns to ignore (e.g., ["*.tmp", ".DS_Store"]).
    """
    # 1. Convert to a Path object and make it an absolute path
    folder_path = Path(folder_path).resolve()

    # 2. Validate the repository type and local path
    if repo_type not in ["model", "dataset", "space"]:
        raise ValueError("Invalid repo_type. Must be 'model', 'dataset', or 'space'.")

    if not folder_path.exists():
        raise FileNotFoundError(f"The provided path '{folder_path}' does not exist.")

    # 3. Initialize the API client
    api = HfApi(token=token)

    # 4. Create the repository if it doesn't already exist
    logger.info("Ensuring repository '%s' exists...", repo_id)
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        private=private,
        exist_ok=True,  # Prevents errors if the repo already exists
    )

    # 5. Upload the local content to the specific revision
    target_branch = revision if revision else "main"
    if folder_path.is_dir():
        logger.info(
            "Uploading directory contents of '%s' to '%s' (branch: %s)...",
            folder_path,
            repo_id,
            target_branch,
        )

        api.upload_folder(
            folder_path=str(
                folder_path
            ),  # Converted to string just in case older HF versions expect it
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            revision=revision,
            commit_message=commit_message,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    elif folder_path.is_file():
        target_file_path = (
            f"{path_in_repo.rstrip('/')}/{folder_path.name}"
            if path_in_repo
            else folder_path.name
        )
        logger.info(
            "Uploading file '%s' to '%s/%s' (branch: %s)...",
            folder_path,
            repo_id,
            target_file_path,
            target_branch,
        )
        api.upload_file(
            path_or_fileobj=str(folder_path),
            path_in_repo=target_file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            token=token,
        )
    else:
        raise ValueError(f"Unsupported upload path type: {folder_path}")

    # 6. Generate the URL to view the uploaded files
    base_url = "https://huggingface.co/"
    prefix = (
        "datasets/"
        if repo_type == "dataset"
        else "spaces/"
        if repo_type == "space"
        else ""
    )
    tree_path = f"/tree/{revision}" if revision else ""

    logger.info(
        "Upload complete! View your files here: %s%s%s%s",
        base_url,
        prefix,
        repo_id,
        tree_path,
    )
