# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cross-Project Context

For an overview of sibling projects and shared conventions, see `/Users/zhiyuanma/Desktop/codes/CLAUDE.md`.

@.agents/CODE_INDEX.md

## Project Overview

flow_grpo_custom — 基于 GRPO + Flow Matching 的 3D 生成强化学习框架。使用奖励模型（PICKScore、法线一致性、Uni3D）引导 Trellis 等 3D 扩散模型，通过蒸馏和对比学习优化生成质量。

## Key References

> `.agents/notes/` 存放调试经验和架构分析，遇到相关问题时应先查阅已有笔记。

## Conventions

- **uv only**: `uv run <command>`, never raw `python`.
- **Temp files → `/tmp/`**: never create scratch files in the project root.
- **Minimal try/except**: let errors propagate.
- **Git**: branch prefixes `feat/`, `fix/`, `chore/`. Draft PRs only.

## Koala GPU Cluster

@.agents/KOALA.md
