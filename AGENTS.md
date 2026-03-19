# AGENTS.md

## Package & Project Management

- **uv** is used for dependency management, virtual environments, and tool installation (not pip/pipx/poetry).
- Run tasks via `uv run <cmd>`. Use the `uv` skill if needed for details.

## Repository Overview

Python replication of [badlogic/pi-mono](https://github.com/badlogic/pi-mono), specifically the **pi-ai** and **pi-agent-core** crates. Naming convention: `pi` → `otter`.

### Key Conventions

- Follow the upstream pi-mono architecture and module boundaries when implementing otter equivalents. This work should be as close as possible to an exact replica written in python.
- Type hints are required; use `pyright` for static checking.
- When referencing upstream code, structure and naming should mirror pi-mono with the `pi` → `otter` substitution.
