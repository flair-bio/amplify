#!/usr/bin/env bash

# Configures repository merge settings AND branch protection
# using the user's local GitHub CLI.

# Exit immediately if a command fails
set -e

echo "⚙️  Starting repository configuration..."

# 1. Check if GitHub CLI (gh) is installed
if ! command -v gh &> /dev/null; then
    echo "❌ Error: GitHub CLI ('gh') is not installed."
    echo "   Please install it first: https://cli.github.com/"
    exit 1
fi

# 2. Check if user is logged into GitHub CLI
if ! gh auth status &> /dev/null; then
    echo "❌ Error: You are not logged into GitHub CLI."
    echo "   Please run the following command and follow the prompts:"
    echo "   gh auth login"
    exit 1
fi


# 3. Get the repo's full name (e.g., owner/repo-name)
REPO_NAME=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "🎯 Targeting repository: $REPO_NAME"


# 4. Apply the merge settings
echo "🚀 Applying merge settings..."
gh repo edit "$REPO_NAME" \
  --enable-merge-commit=false \
  --enable-rebase-merge=true \
  --delete-branch-on-merge=true

echo "✅ Merge settings configured!"

# 5. Define the branch protection rules as a JSON payload
# We use a HEREDOC to make this clean
BODY=$(cat <<'EOM'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "check",
      "test"
    ]
  },
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "enforce_admins": true,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "restrictions": null
}
EOM
)

# 6. Set branch name (default: main, can be overridden by first argument)
BRANCH_NAME="${1:-main}"
echo "🛡️  Applying branch protection rules..."
echo "🔀 Targeting branch: $BRANCH_NAME"

# Run the branch protection and capture output and exit code
echo "$BODY" | gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/$REPO_NAME/branches/$BRANCH_NAME/protection" \
  --input -

echo "✅ '$BRANCH_NAME' branch protection applied!"
echo "🎉 Repository setup complete."
