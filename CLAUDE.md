# video-use — Claude Instructions

Open-source tool to edit videos with Claude Code: drop raw footage in a folder,
chat, get `final.mp4` back. No presets, no menus — works for talking heads,
montages, tutorials, travel, interviews.

## Stack

Python (`pyproject.toml`, package `video-use`). Layout: `helpers/` (the
editing primitives), `skills/` (the Claude-facing skills), `static/` (assets).

## Rules

- **This repository is public.** Never commit absolute local paths, personal
  footage, API keys, or material belonging to any other repository.
- Helpers should stay composable and side-effect-explicit: a helper that writes
  a file says so in its name and docstring.
- Skills are the user-facing contract — changing a skill's trigger wording
  changes when it fires, so treat `description` edits as behaviour changes.
- Commit format: `type: what and why`. Never push without explicit
  confirmation.
