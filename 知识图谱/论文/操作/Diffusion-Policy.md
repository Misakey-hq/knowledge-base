---
title: "Diffusion Policy"
date: "2026-07-25"
aliases:
  - DP
  - 扩散策略
tags:
  - paper
  - robotics
  - imitation-learning
  - diffusion
---

# Diffusion Policy

> **论文**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/)
> **代码**: https://github.com/real-stanford/diffusion_policy
> **arXiv**: 2303.04137v5

## 核心思想

将扩散模型应用于机器人动作生成，通过多次去噪逐步生成合理的动作序列。

## 关键公式

- 策略整体: 观测 → 未来动作序列
- 单次去噪: 带噪动作 + 扩散步 + 观测条件 → 预测噪声
- 实机执行: 二维目标点 → 末端位姿 → 控制器指令

## 笔记

- [[Diffusion-Policy实机Push-T输入输出说明|实机 Push-T 输入输出说明]]
