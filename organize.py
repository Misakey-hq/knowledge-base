#!/usr/bin/env python3
"""
Obsidian 知识库自动整理脚本
扫描收集箱，根据文件内容和元数据自动分类到知识图谱对应目录。
"""

import os
import re
import shutil
import json
from datetime import datetime
from pathlib import Path

BASE = Path("/home/lhq/knowledge-base")
INBOX = BASE / "收集箱"
LOG_FILE = BASE / "organize.log"

# 排除文件（不整理）
SKIP_FILES = {"收集箱.md"}

# 分类规则: { 目标目录: { keywords: [...], extensions?: [...] } }
# 子分类只匹配关键词，不匹配扩展名——避免把不相关的 PDF 吸走
# 只有 "知识图谱/论文" 用 extensions: [".pdf"] 作为兜底（关键词匹配不到的 PDF 放这里）
RULES = {
    "项目": {
        "keywords": ["项目", "project", "计划", "milestone", "进度", "deliverable", "交付"],
    },
    "知识图谱/论文/操作": {
        "keywords": ["操作", "manipulation", "机械臂", "robot arm", "抓取", "grasp",
                      "peg", "assembly", "装配", "push", "dexterous", "灵巧",
                      "imitation learning", "模仿学习", "diffusion policy",
                      "action chunking", "act", "behavior cloning"],
    },
    "知识图谱/论文/导航": {
        "keywords": ["导航", "navigation", "slam", "localization", "定位",
                      "path planning", "路径规划", "mapping", "建图", "lidar",
                      "激光雷达", "里程计", "odometry"],
    },
    "知识图谱/论文/感知": {
        "keywords": ["感知", "perception", "检测", "detection", "分割", "segmentation",
                      "vit", "vision transformer", "cnn", "resnet", "pointnet",
                      "dino", "clip", "sam", "grounding", "yolo", "detr",
                      "自监督", "self-supervised", "ssl", "对比学习",
                      "contrastive", "visual", "视觉", "图像", "image",
                      "特征提取", "feature", "backbone", "预训练", "pretrain"],
    },
    "知识图谱/论文/移动": {
        "keywords": ["移动", "locomotion", "行走", "walking", "四足", "quadruped",
                      "双足", "biped", "腿足", "步态", "gait"],
    },
    "知识图谱/论文/通用方法": {
        "keywords": ["通用方法", "transformer", "强化学习", "reinforcement learning",
                      "diffusion", "扩散", "generative", "生成",
                      "world model", "世界模型", "foundation model", "基础模型",
                      "大模型", "llm", "vlm", "多模态", "multimodal"],
    },
    "知识图谱/论文": {
        "extensions": [".pdf"],  # 兜底：匹配不到任何子分类的 PDF 放这里
    },
    "知识图谱/工具": {
        "keywords": ["工具", "tool", "安装", "install", "配置", "config", "setup",
                      "使用指南", "指南", "guide", "cli", "终端", "terminal"],
    },
    "知识图谱/基础": {
        "keywords": ["基础", "理论", "theory", "数学", "math", "概念", "原理",
                      "fundamental", "概率", "统计", "线性代数", "优化"],
    },
    "知识图谱/实验": {
        "keywords": ["实验", "experiment", "测试", "test", "结果", "result",
                      "benchmark", "评估", "evaluation"],
    },
    "知识图谱/想法": {
        "keywords": ["想法", "idea", "灵感", "思考", "thought", "brainstorm"],
    },
}

# 默认目标（匹配不到时）
DEFAULT_TARGET = "知识图谱"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    # 确保有索引页
    index = path / f"{path.name}.md"
    if not index.exists():
        index.write_text(f"# {path.name}\n", encoding="utf-8")


def classify(file: Path) -> str | None:
    """根据文件名、内容和扩展名判断分类，返回目标目录（相对于BASE）"""
    ext = file.suffix.lower()
    name = file.stem.lower()
    text = ""

    if ext == ".md":
        try:
            content = file.read_text(encoding="utf-8")
            # 提取 frontmatter tags
            fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if fm_match:
                fm = fm_match.group(1)
                text += fm + "\n"
            text += content[:3000]  # 取前3000字符分析
        except Exception:
            pass

    # 合并搜索文本: 文件名 + 内容，统一分隔符为空格
    search_text = f"{name} {text}".lower()
    search_text = search_text.replace("_", " ").replace("-", " ")

    scores = {}  # {target: score}

    for target, rule in RULES.items():
        score = 0
        # 扩展名匹配（弱信号，作为兜底）
        if "extensions" in rule and ext in rule["extensions"]:
            score += 1
        # 关键词匹配（强信号，每个关键词 +5）
        for kw in rule.get("keywords", []):
            if kw.lower() in search_text:
                score += 5
        if score > 0:
            scores[target] = score

    if not scores:
        return None

    # 选得分最高的；同分则选路径更具体的（层级更深的）
    best = max(scores, key=lambda t: (scores[t], t.count("/")))
    return best


def organize():
    log("========== 开始整理收集箱 ==========")
    files = [f for f in INBOX.iterdir() if f.is_file() and f.name not in SKIP_FILES]

    if not files:
        log("收集箱为空，无需整理。")
        return

    moved = 0
    for file in files:
        target_dir_name = classify(file)
        if target_dir_name is None:
            log(f"⏭ 无法分类，保留在收集箱: {file.name}")
            continue

        target_dir = BASE / target_dir_name
        ensure_dir(target_dir)

        # 避免重名
        dest = target_dir / file.name
        if dest.exists():
            stem, ext = file.stem, file.suffix
            dest = target_dir / f"{stem}_{datetime.now().strftime('%Y%m%d')}{ext}"

        shutil.move(str(file), str(dest))
        log(f"✅ {file.name} → {target_dir_name}/")
        moved += 1

    log(f"整理完成，移动 {moved}/{len(files)} 个文件。")


if __name__ == "__main__":
    organize()
