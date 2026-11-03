# 分割模块与完整链路对比合并报告

版本：代码 v0.4.0；接口 pipeline-1.0；日期：2026-10-26。

已把团队提供的分割输出接入已有 DINO → SAM2.1 → ResNet-18 模块化链路。既可以读取 segmentation.json 直接继续材质推理，也可以从原始 RGB 运行完整链路。两条真实路径均输出 34 个实例，文件与引用校验通过；深度仍仅保留引用和状态。此次没有训练新模型，也没有把自动预测转成训练真值。

## 一、收到的包与原有链路

| 项目 | 团队提供的包 | 合并前链路 | 合并处理 |
|---|---|---|---|
| 分割输出 | pipeline-1.0 segmentation.json、report.md、34 张 mask、overlay | result.json 为分割原始记录；perception.json 为聚合结果 | 正式输出统一为五阶段 pipeline-1.0，兼容旧聚合格式 |
| 实例字段 | object_label、mask_path、整数 bbox、det_score、seg_score=null | label、mask_ref、浮点检测框 | 映射字段，统一从 mask 重算整数半开框 |
| 关联标识 | run_id/frame_id/instance_id | 实例按检测排序编号 | 导入保留上游标识；不同运行按空间 mask 匹配，仅用于对比 |
| 材质 | 未运行 | ResNet-18 五类候选 | 对新包原始 RGB 和 mask 框重新推理，不复制旧编号下的预测 |
| 深度 | 路径、1280×960、uint16，单位与配准未确认 | 可选已配准深度统计 | 默认 reference；复制已找到的同帧深度，depth_usable=false |
| 模型比较 | mask_compare.zip 有 4 张整帧图和 24 张局部图 | 历史选型报告与脚本 | 作为历史视觉证据保留，不生成新的模型精度排名 |

frame_211.zip 不包含原始 RGB 和深度文件，preprocess_ref=null。已使用工作区中同名帧的原始 RGB 与深度完成联调；原包无 RGB 哈希，导入器只能自动核验尺寸，同帧身份仍由调用者提供。新结果记录实际 RGB 哈希以及输入分割文件哈希。原有相对深度引用在单独解压后失效，通过显式 --depth 指定现有同帧文件后已复制到合并结果中。

## 二、逐实例对比结论

对照来源：团队 frame_211/segmentation.json 与 2026-10-23 material_v2_pipeline 同帧 perception.json。按 mask IoU 降序贪心一对一匹配，阈值 0.5，不依赖序号，未使用人工真值。

| 指标 | 本次核验值 |
|---|---:|
| 图像尺寸 | 1280×960 |
| 双方实例数 | 34 / 34 |
| 成功空间匹配 | 34 |
| 未匹配实例 | 0 / 0 |
| 平均预测 mask IoU | 0.998805 |
| 最低预测 mask IoU | 0.991974 |
| 同一对象编号不同 | 33 / 34 |
| 原样保留的上游 mask 像素 | 34 / 34 |

结论是两次分割高度一致，不能据此报告准确率。33 个对象的编号不同，直接按 i000、i001 拼接旧材质会错配。完整对应表保存在 instance_mapping.csv 与 comparison_merged_vs_previous.json。本次从 RGB 重跑的新链路同样得到 34 个实例，与团队包的平均 mask IoU 为 0.998805。

新包 32 个 packet、2 个 carton。视觉复核发现 i026、i030 覆盖两侧设备部件，应列为疑似背景误检，待人工确认；当前保留原始输出，没有为该帧单独调 Prompt、阈值或静默删除实例。

## 三、材质衔接与状态

复用 2026-10-23 candidate.pt，SHA-256 为 `75fd02d43a7eacedd926090eda7e1b5acf3c953510b2d847869ee4f3266c8918`。输入是原始 RGB 的实例 bbox 裁剪，不使用 overlay；默认保留原背景，预处理沿用当前 ResNet-18。34 个实例全部返回模型候选：metal 2 个、paper 13 个、plastic 18 个、wood 1 个。这些是模型输出分布，不是人工确认材质，也不是准确率。

相对旧链路，空间匹配后有 3 个实例的 top-1 不同。新链路由最终 mask 重算裁剪框，旧链路用浮点检测框；同时两次 mask 有微小差别，因此本次并非严格单变量精度实验，也不据此判断新旧优劣。

| 新包编号 | 旧链路编号 | 原候选 | 本次候选 |
|---|---|---|---|
| i011 | i005 | paper | plastic |
| i029 | i025 | plastic | metal |
| i025 | i020 | paper | wood |

当前没有自动接触面选择和经验证的接受/拒识策略。material_input.json 明示 `selection_method=instance_bbox_fallback`、`contact_verified=false`；正式 material_result.json 保留 top1_label/score/scores，同时 `label=unknown`、`status=unknown`、`reason=decision_policy_not_validated`。因此分割状态为 ok，最终状态为 partial，表示工程链路完成但材质结果尚未达到正式接收条件。unknown 不代表模型学到了第六类。兼容文件 perception.json 的 material.label 仍是原始 top-1。

## 四、深度与历史模型对比的边界

深度文件存在且可以读出 uint16、1280×960；这不能证明配准或单位已确认。单位和 alignment 均保留 unconfirmed，depth_scale_to_m=null，depth_usable=false，processing_status=reference_only。此次不执行等高线、深度尺寸、点云或三维几何计算。旧可选统计保留在显式 statistics 模式下，正式契约不把它标为深度验收完成。

mask_compare.zip 共 28 张 JPEG，没有逐实例原始双模型 mask、运行配置或指标 JSON。其文件名和颜色与现有 compare_masks.py 对应：红为 SAM1 独有、蓝为 SAM2.1 独有、绿为交集；这是依据现有脚本对图例的解释，原 ZIP 未提供生成清单。该脚本历史实现按列表位置配对，需要警惕实例顺序变化。本次不从 JPEG 反推 mask 像素，也不重新计算历史所报 0.948。该历史数只能作为报告引用；预测间 IoU 不等于真值精度。

原报告约 19.8 秒是上游记载的 CPU 推理时间。旧 GPU 链路、本次冷启动运行的硬件、软件环境与计时范围不同，不直接计算提速倍数。本次 timing 明确不含契约文件导出，不作为完整端到端性能基准。后续速度验收应固定输入、设备、预热与计时边界后重复测量。

## 五、交付内容与接口兼容

```text
code/                       完整模块化源码 v0.4（不含环境、模型权重）
evidence/source/frame_211/   原始分割包解压内容
evidence/source/mask_compare/历史 28 张对比图片
evidence/merged_frame_211/   读取上游分割后继续下游的真实结果
evidence/fresh_full_pipeline/从原始 RGB 重跑的真实结果
evidence/*comparison*.json  两次结果的空间匹配和一致性
evidence/instance_mapping.csv  新旧实例编号及材质候选映射
evidence/merge_audit.json    输入哈希、候选变化和测试证据
```

每帧的正式文件：preprocess.json → segmentation.json → material_input.json → material_result.json → result.json，另含 report.md、masks/*.png、rois/*、rgb/frame.png、可选 depth/* 和 overlay.jpg。五阶段共用 run_id/frame_id，各实例沿用 instance_id。

兼容变化：最终 result.json 现在是 pipeline-1.0，旧分割原始记录改名 segmentation_raw.json；旧聚合 perception.json 保留。调用者若依赖旧 result.json，需切换到 segmentation_raw.json 或适配新结构。mask 框替代检测框也会改变材质裁剪区域。批处理 status=ok 表示程序运行成功，pipeline_status=partial 表示材质仍未正式接受，应分别读取。

在源码目录执行（先按 perception/README.md 安装环境并准备材质权重）：

```powershell
python -m perception.pipeline.run --segmentation ../evidence/source/frame_211/segmentation.json --rgb ../evidence/merged_frame_211/rgb/frame.png --depth ../evidence/merged_frame_211/depth/2026_08_26_09_17_12_211_depth.png --material-weight path/to/candidate.pt --device cpu --output output/resumed
python scripts/validate_pipeline_bundle.py output/resumed
```

导入入口仅需材质权重，不加载 DINO/SAM；完整 RGB 重跑还需按模型清单准备 DINO/SAM 权重和配置。输出目录必须为空。

## 六、验证与后续事项

33 项主测试、4 项历史分割测试，共 37 项全部通过。新增覆盖整数框、跨阶段关联、ID 保留、重排匹配、错误尺寸、非二值/空 mask、不安全路径、异常分数、零目标、单实例材质失败后继续和未知深度引用。两个真实结果包共 68 个实例均通过 mask、框、RGB 尺寸、ROI 文件/尺寸、材质分数归一化、阶段引用和状态传播检查；上游 34 张 mask 像素全部未变。

下一版本优先：补原始 RGB 哈希和 preprocessing 关联、人工复核设备误检与 mask、实现接触面区域选择、在独立验证集确定材质拒识策略、确认深度单位/配准/无效值/相机内参后实现等高线和尺寸。模型权重和企业样例在本地使用；此次未提交、推送或修改远程仓库。
