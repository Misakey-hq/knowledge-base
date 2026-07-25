#!/bin/bash
# Obsidian 知识库自动同步脚本
# 每天：先整理收集箱 → 再提交推送

cd /home/lhq/knowledge-base

# 1. 自动整理收集箱
python3 organize.py

# 2. 如果没有变更，直接退出
if git diff --quiet && git diff --cached --quiet; then
    echo "[$(date '+%Y-%m-%d %H:%M')] 无变更，跳过。"
    exit 0
fi

# 3. 提交并推送
git add -A
git commit -m "chore: 每日自动同步 $(date '+%Y-%m-%d')"
git push origin master

echo "[$(date '+%Y-%m-%d %H:%M')] 同步完成。"
