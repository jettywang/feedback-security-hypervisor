# Hyper-V 配置世代选择取证审计

对象：`dissect/hypervisor/descriptor/hyperv.py` 中的 `HyperVFile`（底层容器 `HyperVStorage`，
版本 `0x400`，即 `.vmcx` / `.VMRS` 的注册表式二进制格式）。

本目录提供可执行的诊断材料，回答三个问题：

1. 最高序号的表 / header 是否一定等于“最后一个一致状态”？
2. 当前实现实际暴露的是哪一代配置？
3. “已经回放恢复”与“仅收集了日志”如何区分？

## 结论（先给答案）

**最高序号不代表最后一个一致状态。** 序号（`sequence_number`）只表示“最后一次被写入的
候选结构”，写入可能只完成了一半。当前代码把序号当作一致性的充分条件，没有任何一道校验来
证明高序号结构是完整的：

- 双 header：`hyperv.py:55`
  `self.header = header1 if header1.sequence_number > header2.sequence_number else header2`
  纯比序号；不校验两个 header 的 CRC32，也不检查被选中 header 的签名是否有效。
- key table：`hyperv.py:87` 按 `sequence_number` 逆序排序，`hyperv.py:100` 直接取 `[0]`；
  不校验表头 CRC，也不检测条目区撕裂。
- replay log：`hyperv.py:65` 与 `hyperv.py:142` 只**解析并收集** `HyperVStorageReplayLogEntry`，
  类文档（`hyperv.py:31-33`）明确写明 “replaying ... is not currently implemented”。
  没有任何代码把日志条目应用回 key table / file object。

因此，`HyperVFile` 暴露的是 **“按最高序号挑选出来的、磁盘上当前这批 key table 的原样内容
（pre-replay 镜像）”**，而不是“回放后恢复出的一致状态”。在脏文件（replay log 有未结清条目）
上，它给出的是事务应用**之前**的旧镜像；在撕裂文件上，它可能直接报错，或静默返回被零填充的
伪键（见 S4）。只有当所有选中结构 CRC 通过、父子引用全部可解析、且 replay log 为空时，才能把
当前镜像称为已证明一致的配置——这也正是独立审计器 `hive_audit.py` 的判定口径。

## 关键格式事实（已对随仓真实样本验证）

- 所有 header / 表 / 对象表项的 checksum 都是 **CRC32，校验时把 checksum 字段清零**。
- storage header（46 字节）与 replay log header（34 字节）的 CRC 覆盖整个头部。
- key table 的 checksum **只覆盖 10 字节表头，不含条目体**；条目另有自己的 checksum 字段，
  但真实文件那里用的是 VmDataStore 专用算法（不是普通 CRC32），所以不能拿普通 CRC32 去否决条目。
- 对象表固定在 `0x2000`，其表项 `size` 是**页对齐**的（恒为整页），因此无法用它发现页内撕裂；
  撕裂要靠条目结构自相矛盾（部分头、零头槽、越界 size、不可能的 `data_offset`、有头无体）来识别。
- 跨代父子引用 = 条目头里的 `(parent_table_idx, parent_offset)`；父项只在被选中的那代表里
  解析（`hyperv.py:283-291`），指向旧代表里偏移时会在新代表里查不到。
- `data_offset` 相对的是“去掉 21 字节条目头之后的条目体”，键名从体偏移 0 开始（与库的
  `key` 切片 `raw[:data_offset-1]` 一致）。

## 五个最小复现场景

生成器 `hive_build.py` 只依赖标准库并自带 CRC32；驱动 `reproduce.py` 同时跑库行为与独立审计器。

```
.venv/bin/python forensics/hyperv/reproduce.py
.venv/bin/python forensics/hyperv/hive_audit.py <任意 .vmcx/.VMRS/合成样本>
.venv/bin/python -m pytest tests/descriptor/test_hyperv_generation_forensics.py -q
```

| 场景 | 构造 | HyperVFile 实际行为 | 审计器结论 |
|---|---|---|---|
| S1 同键跨代不同值 | 表 idx=1 有 seq=1(answer=111) 与 seq=2(answer=222) | 按序号选 seq=2，返回 `answer=222` | `True`（两代各自完整、日志空，可证明） |
| S2 子项引用跨代父项 | 新代表的 child 指向只存在于旧代表的父节点偏移 | 构造期 `KeyError`，整树无法链接 | `False`，dangling parent |
| S3 较新 header 不完整 | 高序号 header 签名/CRC 损坏，旧 header 完好 | `InvalidSignature` 直接加载失败 | `False`，选中的 header CRC 无效 |
| S4 较新 key table 撕裂 | 高序号表被截断，旧代表完好 | **仍加载并选撕裂表**，静默返回 16 个 `\x00` 的伪根键 | `False`，`torn=True`（表头 CRC 仍为 True，证明单靠表头 CRC 不够） |
| S5 脏 replay log | 结构完整但有 3 条未结清日志 | 收集到 3 条日志但**从不回放**，返回回放前镜像 answer=111 | `None`（无法证明，不得称“最新有效配置”） |

## “已回放恢复” vs “仅收集日志” 的判定边界

- **仅收集日志（当前实现的真实能力）**：`replay_logs[*].entries` 被解析出来，`num_entries > 0`
  即可见，但 key table / file object 的字节没有任何改变。S5 中库返回的 `answer=111` 是日志事务
  应用**之前**的值。
- **已回放恢复（当前实现不具备）**：需要逐条按 `(offset,size,data_checksum)` 校验并把日志数据
  写回对应区域、推进世代、清空/归档日志后，才能得到新的一致镜像。代码里不存在这一步。
- 因此审计器采用三态：`True`（结构 CRC 全过 + 父引用全解析 + 日志空）、`False`（存在硬伤：
  CRC 失败 / 撕裂 / 悬空引用）、`None`（结构没坏但日志未结清，一致性在真正回放前无法证明）。
  **`None` 与 `False` 都不允许在结论里写成“最新有效配置”。**

## 风险边界（不要过度外推）

- 合成样本是为触发特定选择路径而构造的最小结构，布局（固定页、单一对象表）比真实文件简单；
  但所有魔数、结构尺寸、CRC 算法、`data_offset` 语义均已对照随仓真实样本校准。
- 审计器能证明“硬伤”和“日志未结清”，但**不能**在缺少回放实现的情况下证明脏文件回放后的最终值；
  它刻意对这种情况返回 `None` 而不是猜测。
- 条目级 checksum 的真实算法未在本审计中逆向，故条目完整性依赖结构合理性 + 表头 CRC，
  不对条目 checksum 做普通 CRC32 否决（否则会误报真实文件）。
- 本审计只读取与生成诊断文件，**未修改 `dissect/` 下任何库代码**，现有测试保持通过。
