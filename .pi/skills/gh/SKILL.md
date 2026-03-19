---
name: gh
description: Use this skill when interacting with  the GitHub repository, managing issues, or performing any GitHub CLI operations. Use when queries mention GitHub issues, pull requests, repositories, labels, or the gh CLI tool.
---

# GitHub CLI (gh) Skill

## Overview

Use the `gh` CLI tool for all GitHub repository interactions, including issue management, pull requests, and repository operations.

## Issue Management

Use `gh issue` commands to create, list, and manage issues.

### Required Labels

Every issue **must** be labeled with one or more of the following tags:

| Label | Description |
|-------|-------------|
| `feature` | A feature that does not currently exist in the repository |
| `bug` | A defect or incorrect behavior |
| `alignment` | A mismatch between this repository and the upstream [pi-mono](https://github.com/badlogic/pi-mono) source |

### Referencing Upstream Files

All issues **must** reference the relevant pi-mono source files they need to replicate. This provides clear traceability between the otter-mono implementation and the upstream pi-mono architecture.

When creating or editing an issue, include a section like:

```markdown
## Upstream Reference

- `pi-ai/src/path/to/file.rs` → replicated as `otter-ai/src/path/to/file.py`
```

### Commits and Closing Issues

When work has been completed the issue should be mentioned in the commit and closed automatically.
