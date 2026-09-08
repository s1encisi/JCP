# JCP · ESRL-CMO

铜电积代理建模、安全强化学习与约束多目标优化项目的私密源码仓库。

## 上传范围

本仓库通过 `.gitignore` 中的逐文件白名单，仅跟踪已检查的源码、依赖清单和本说明。未列入白名单的新文件默认忽略，包括放在源码目录里的数据文件。

| 目录 | 跟踪内容 |
| --- | --- |
| `src/esrlcmo_project/interface/` | Web 界面、Flask 后端、优化算法及依赖清单 |
| `src/esrlcmo_project/modeling/code/` | 预处理、代理建模、基准实验与模型解释代码 |
| `src/esrlcmo_project/optimization/` | NSGA-II、PPO-Lagrangian、约束 MOEA、验证与分析代码 |
| `src/esrlcmo_project/analyses/` | 时序验证、稳健性及归档结果分析代码 |
| `src/revision/` | 修订阶段的算法对比、经济分析、稳健性、时序验证与绘图脚本 |

以下材料仅保留在本地，不进入仓库：

- 全部 Excel、CSV、TSV、JSON、数据库及其他数据文件。
- 训练好的模型、策略权重、实验结果、图表与缓存。
- 论文、补充材料、审稿意见、回复信及投稿材料。
- 历史版本、服务器提交脚本、部署说明和包含论文正文的文档生成脚本。
- 环境变量文件、凭据、密钥以及其他未单独审核的文件。

忽略规则不会删除或移动本地文件。私密仓库用于源码版本管理，不是完整研究资料备份。

## 本地运行

界面入口为 `src/esrlcmo_project/interface/app_v2.py`，依赖清单位于同目录的 `requirements.txt`。在项目根目录可执行：

```powershell
python src/esrlcmo_project/interface/app_v2.py
```

界面默认使用 `http://127.0.0.1:5001`。训练、推理和结果分析还需要本地的数据、模型权重与结果模板，这些材料不随克隆提供。部分修订脚本沿用历史目录路径，执行前需结合当前本地布局核对；本次仓库初始化未修改科研代码或运行实验。

## 后续提交

已有白名单文件的改动可以正常提交。新增源码应先检查是否包含真实数据、账号、服务器地址或凭据，再将其准确路径加入 `.gitignore` 的白名单区。

提交前核对实际暂存内容：

```powershell
git status --short
git diff --cached --name-only
git diff --cached
```

请勿用 `git add -f` 将数据或保密文件强制加入仓库。已跟踪文件的内容变更仍需审核，`.gitignore` 不会检查源码中后来写入的数据或凭据。
