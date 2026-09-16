# 运行与演示说明

## 公开合成演示

在项目根目录使用 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe run_demo.py
```

打开 http://127.0.0.1:5001/。顶部 `SYNTHETIC DEMO` 表示合成模式，不需要任何模型密钥、数据库连接字符串或研究权重。

修改端口与运行目录：

```powershell
$env:JCP_PORT = '5002'
$env:JCP_RUNTIME_DIR = Join-Path (Get-Location) '.runtime\my-demo'
.\.venv\Scripts\python.exe run_demo.py
```

演示状态默认保存在 `.runtime/demo/`。更换运行目录即可得到独立的空白演示，不必覆盖已有记录。停止服务使用运行终端中的 Ctrl+C。

## 录入到导出的完整操作

1. 打开 Data Management，选定工况并输入数值。可使用进液Cu=38、温度58、电流18000、流量117、时长4的Condition 1合成场景。
2. Submit Data 写入经过验证的记录；Predict Now 将这些输入传入预测表单。
3. Run Prediction 后查看来源和五项约束。Apply to Overview 将本次预测显示到主页。
4. 打开 NSGA-II Optimization。默认读取当前工况最新的本地记录，没有记录时使用明确的合成默认值。
5. 默认64个个体、30代。完成后查看实际生成的二维投影与四类代表方案。
6. 点击 Apply 后，主页的工况、输入、目标和约束全部来自所选方案。相对变化使用同一次优化的输入基准。
7. 打开 Experiment Records，选择记录并导出 CSV。文件包含实际候选解，不是固定模板。

文件导入可先生成样例：

```powershell
.\.venv\Scripts\python.exe examples/create_demo_data.py
```

导入 `.runtime/synthetic_import.csv`。CSV/XLSX需要 `cu_in, temperature, current, flow, duration, mode` 六列；模式为 `three_stage`、`four_stage`、`serial`。单行导入的串联记录表示两个单元使用同一组T/I/Q；独立设置两个单元时使用预测表单。

导入不会静默填充缺失字段。一个批次中发现无效行时整批拒绝，修正后重新导入。Validate Stored Records 只检查已有记录，不在后台执行未声明的插补或修改训练集。

## 本地研究模型

研究模式需要与训练环境兼容的 scikit-learn、joblib、pandas、NumPy 和 PyTorch 等依赖。原研究环境记录位于 `src/esrlcmo_project/optimization/requirements.txt`；正式训练还应根据所用建模/解释脚本安装相应依赖。

只加载自己训练或来源可信的模型文件。joblib属于序列化模型格式，可信来源是本地加载的前提。

```powershell
$env:JCP_DEMO = '0'
$env:JCP_MODEL_DIR = Join-Path (Get-Location) 'src\esrlcmo_project\modeling\outputs'
$env:JCP_POLICY_DIR = Join-Path (Get-Location) 'src\esrlcmo_project\interface\pt'
python src/esrlcmo_project/interface/app_v2.py
```

模型目录包含 `cu_three_stage`、`as_three_stage`、`voltage_three_stage`、`cu_four_stage`、`as_four_stage`、`voltage_four_stage`、`cu_serial`、`as_serial`、`voltage_serial` 九个子目录，各自使用对应的 `extra_trees_<工况>.joblib` 文件名。

策略目录使用 `ppo_actor_condition1.pt`、`ppo_actor_condition2.pt`、`ppo_actor_condition3.pt`。权重须与代码中的128单元网络结构、状态顺序和动作缩放一致。加载失败明确显示不可用，不返回模拟策略结果。

预测页的 Evaluate Local PPO-Lagrangian Policy 调用保存策略，按增量动作解码，再重新预测并检查五项约束。进液Cu保持外部输入，界面不向设备下发操作命令。

正式训练与研究优化通过原研究CLI运行。先查看对应脚本的 `--help`，核对资产、目录和计算预算。研究脚本的 `--fast` 也可能包含较大训练预算，不能将这个名称理解为几秒钟的演示。

## 常见情况

| 现象 | 处理 |
| --- | --- |
| 5001端口占用 | 设置 `JCP_PORT` 使用其他本地端口 |
| 研究模型不可用 | 检查模型目录与特征约定，或用 `run_demo.py` 明确运行合成演示 |
| 409提示优化正在运行 | 等待当前任务完成，避免重复启动 |
| 输入返回422 | 检查工况、字段、单位、范围和非数值内容 |
| 训练按钮不可用 | 正式训练使用本地CLI，网页不会模拟“训练完成” |
| 原始文件不能通过URL下载 | 属于预期行为，仅生成的CSV使用受限下载路径 |

## 后续提交

仓库采用逐文件白名单。新增源码和文档先审核，再加入 `.gitignore`。启用本地提交检查：

```powershell
git config core.hooksPath .githooks
python tools/audit_publication.py
```

自动检查不能判断每一段业务信息是否保密。企业标识、真实样本、模型资产、截图和实验附件仍需人工确认公开范围。
