# Repository working agreement

## Project and sources

- This is `Catpockets/sdm-vision-reliability`, a Berkeley MIDS research project on Similarity-Distance-Magnitude (SDM) uncertainty estimation for vision.
- Read `README.md`, `docs/project-brief.md`, and `docs/allen-schmaltz-sdm.md` before changing research scope or methodology. Preserve source attribution and `docs/assets/poster.png`.
- As of this setup, the repository contains documentation only. Python, PyTorch, and Jupyter are planned; there is no dependency manifest, runnable pipeline, test suite, or GitHub Actions workflow yet. Reinspect the tree before assuming this remains true.
- Core datasets are CIFAR-100, CIFAR-100-C, and SVHN. Vision-language experiments are optional future scope.
- Treat 95% accuracy on admitted predictions as a research target. Report coverage and uncertainty alongside accuracy; state assumptions and do not imply universal guarantees under distribution shift.
- Keep training, calibration, and held-out evaluation separate. Record seeds, versions, dataset provenance, and reproduction commands when adding experiments. Fit calibration and choose thresholds without using held-out test data.

## Ownership and authorization

- Act as the primary engineer: investigate, implement, validate, and prepare reviewable work. The human lead tests, approves, and merges. **Never merge a pull request.**
- Routine authenticated GitHub CLI inspection, task issue creation/updates, parent and milestone assignment, task branch pushes, and task PR creation/updates are authorized. Do not ask again for routine delivery.
- Do not force-push, rewrite history, deploy, publish packages, create releases, submit to stores, or close/delete unrelated issues without explicit authorization for that action.
- Preserve unrelated local changes and existing branches/PRs. Do not recycle closed issue numbers or reopen abandoned scope silently.
- Never print authentication tokens, commit credentials, or include authentication output in issue or PR bodies. If authentication fails, report the blocker; do not extract tokens from local files.
- Follow more-specific nested `AGENTS.md` guidance where present, subject to the user's instructions.

## GitHub CLI workflow

Use `gh` for GitHub operations and `git` for local history and pushes. Run commands from the repository root. Examples use Bash variables; replace example titles and bodies with the actual task. Consult `gh <command> --help` for installed CLI options.

### 1. Inspect local and live state

```bash
repo=Catpockets/sdm-vision-reliability
git status --short --branch
git remote -v
gh repo view "$repo" --json nameWithOwner,defaultBranchRef,url
gh issue list --repo "$repo" --state all --limit 100 --json number,title,state,labels,milestone
gh label list --repo "$repo"
gh api --paginate "repos/$repo/milestones?state=all"
gh pr list --repo "$repo" --state open --json number,title,headRefName,baseRefName,url
gh workflow list --repo "$repo"
```

Read relevant issue bodies with `gh issue view NUMBER --repo "$repo"`. Increase limits or paginate if necessary. Inspect epics and native parent/sub-issue relationships before organizing new work; the following query inspects the first 100 open issues (paginate if `hasNextPage` is true):

```bash
gh api graphql -f query='{
  repository(owner: "Catpockets", name: "sdm-vision-reliability") {
    issues(first: 100, states: OPEN) {
      nodes { number title parent { number title } subIssues(first: 1) { totalCount } }
      pageInfo { hasNextPage endCursor }
    }
  }
}'
```

At setup, the default branch is `main`, tasks use `documentation` or `enhancement` labels, and no milestones or epic hierarchy exist. Verify live state each time. Related starter work: #3 methodology, #4 environment, #5 datasets, #6 baselines. This setup supplies ignore rules for #4; it does not complete environment setup.

### 2. Use an active issue before editing

Every repository commit must belong to an active issue with a user story and observable acceptance criteria. Reuse an open issue only when its scope matches. Otherwise write the task body to a temporary file outside the repository, then create the issue:

```bash
issue_body=$(mktemp)
cat > "$issue_body" <<'EOF'
## User story

As a contributor, I want [specific outcome] so that [benefit].

## Acceptance criteria

- [ ] [Observable behavior and validation requirement]

## Related work

[Dependencies and scope boundaries, if applicable]
EOF
gh issue create --repo "$repo" --title 'Task-specific title' \
  --label documentation --body-file "$issue_body"
rm "$issue_body"
```

Choose the existing label that fits. If native sub-issues are in use, assign exactly one owning epic and its matching milestone; current CLI supports `gh issue create --parent NUMBER --milestone 'TITLE'`. Check installed help before using these flags. Record cross-epic dependencies in the body rather than assigning multiple owners. Do not use an epic as a catch-all task or close it before child outcomes are complete.

### 3. Create a dedicated task branch

Set `issue` to the actual active issue number and choose a descriptive slug. Each file-changing task needs its own short-lived remote branch, including documentation and maintenance work.

```bash
issue=123 # Replace with the actual active issue number.
default_branch=$(gh repo view "$repo" --json defaultBranchRef --jq '.defaultBranchRef.name')
branch="docs/${issue}-descriptive-slug"
git fetch origin
git switch -c "$branch" "origin/$default_branch"
git status --short --branch
```

Use `feature/`, `fix/`, or `docs/` as appropriate. Inspect the current branch before editing; never implement or commit on the default branch. If unrelated changes prevent switching safely, use an isolated worktree. Do not reset, discard, or silently stash someone else's work. Include the issue number in any future repository-required build identity.

### 4. Validate, commit, and push

- Review the complete branch diff and stage only task files by name.
- For documentation, check local links, factual consistency, and `git diff --check`. For ignore changes, use `git check-ignore --no-index` on representative paths that should be ignored and paths that must remain trackable.
- No automated application tests currently exist. Do not claim a test suite passed. When implementation adds documented checks, run the applicable checks and report exact commands and results. Do not install an arbitrary stack merely to validate documentation.

```bash
git diff --check
git diff "origin/$default_branch" --
git add path/to/task-file # Replace with the actual task files.
git diff --cached
git commit -m "docs: describe the task (#$issue)"
git push -u origin "$branch"
```

Use normal additional commits for follow-up fixes; do not amend published commits or force-push.

### 5. Create or update exactly one task PR

Check for an existing open PR from this task branch before creating one:

```bash
gh pr list --repo "$repo" --state open --head "$branch" --base "$default_branch"
```

Write the full PR description into a temporary file and pass `--body-file`. Use real newlines and a quoted heredoc for literal content; never interpolate untrusted prose into shell code. Include:

```markdown
## Summary

- Complete branch changes and user-visible behavior.
- Relevant architecture, privacy, accessibility, migration, or compatibility notes.

## Validation

- `exact command` — truthful result, including failures or checks not run.

## Lead test and acceptance checklist

- [ ] Check out the task branch and follow the documented setup.
- [ ] Exercise the specific primary acceptance behavior.
- [ ] Check relevant regressions and target platforms.
- [ ] Confirm acceptance criteria and approve the pull request.

Issue: https://github.com/Catpockets/sdm-vision-reliability/issues/123

Closes #123
```

Replace placeholders with concrete steps and the real issue number. Leave human-only checks unchecked. End the body with the auto-close keyword, and include the full clickable issue URL above it.

```bash
# Set pr_body to the temporary file containing the completed description.
gh pr create --repo "$repo" --base "$default_branch" --head "$branch" \
  --title 'Task-specific title' --body-file "$pr_body"
# For an existing task PR, use this instead of creating another:
gh pr edit NUMBER --repo "$repo" --body-file "$pr_body"
```

Verify the PR's base, exact head commit, remote branch, and checks:

```bash
gh pr view NUMBER --repo "$repo" --json url,state,baseRefName,headRefName,headRefOid
git rev-parse HEAD
git ls-remote --heads origin "$branch"
gh pr checks NUMBER --repo "$repo"
git status --short --branch
```

The local commit, remote branch commit, and PR `headRefOid` must match. An absence of configured checks is not a passing CI run. Deliver the PR for human review; never run `gh pr merge`.

## Files and handoff

- Keep local environments, secrets, downloaded datasets, checkpoints, and generated experiment artifacts out of Git. `.gitignore` reserves root `data/`, `datasets/`, `checkpoints/`, `outputs/`, `runs/`, `logs/`, `wandb/`, `mlruns/`, and `artifacts/` for local/generated files.
- Keep source notebooks, dependency manifests and lockfiles, configuration, small reviewed fixtures, and documentation assets trackable. Ignore rules do not remove files already tracked; review staged changes explicitly.
- Use `.env.example`, `.env.sample`, or `.env.template` only for sanitized templates containing no real credentials. Review new storage locations before committing large files.
- Finish with the completed result, validation and limitations, commit hash, push result, PR URL, and task branch. Include a copyable `git fetch origin` and `git switch --track origin/<task-branch>` command (or `git switch <task-branch>` if it already exists). Clearly separate automated checks from the lead's outstanding visual, device, or product acceptance checks.
