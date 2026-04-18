# 手相双 Agent + 高级 RAG 工程

本工程是一个可运行的“看手相”大模型应用骨架，目标是把你的思路直接工程化：

- Agent 1（视觉感知）: 图像预检 -> 置信度分流 -> 结构化特征提取
- Advanced RAG（知识引擎）: Query 改写 + HyDE + 混合检索 + Graph 检索 + Re-rank
- Agent 2（命理解读）: 首次解盘 + 多轮追问 + 自检反思

## 1. 目录结构

- app/api: FastAPI 路由
- app/agents: VisionAgent、InterpreterAgent
- app/rag: 改写、HyDE、混检、图检索、重排序
- app/services: LLM/CV/Prompt/Session 服务
- prompts: 全中文提示词
- knowledge: 全中文知识库（规则 + QA + 图谱边）
- storage/sessions.json: 会话持久化

## 2. 技术点映射

1. 本地图像预检（非手相/模糊/可用）
2. OpenCV 边缘增强辅助掌纹识别
3. Query Rewriting（结合 profile + history）
4. HyDE 假设文档生成
5. Hybrid Retrieval（TF-IDF Dense + BM25 + Graph）
6. Re-ranking（CrossEncoder 可选，失败自动降级词法）
7. Self-Reflection（输出前自检修订）

## 3. 快速启动

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 配置模型

把 `.env.example` 复制为 `.env` 后填写（或直接写入系统环境变量）：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_TEXT_MODEL`
- `LLM_VISION_MODEL`
- `LLM_SUMMARY_MODEL`
- `LLM_REWRITE_MODEL`
- `LLM_REFLECT_MODEL`

说明：
- 如果未配置模型，系统会自动走降级逻辑（仍可跑通流程，但视觉抽取和生成质量会下降）。
- 程序会自动读取项目根目录 `.env` 文件。

推荐模型角色分配：

- `LLM_VISION_MODEL`: 图像理解模型（提取掌纹客观特征）
- `LLM_SUMMARY_MODEL`: 首轮解盘总结模型（可用较小模型）
- `LLM_TEXT_MODEL`: 多轮追问最终答复模型（可用联网大模型）
- `LLM_REWRITE_MODEL`: Query 改写模型（可用较小模型）
- `LLM_REFLECT_MODEL`: 反思修订模型（可用较小模型）

### 3.3 启动服务

```bash
uvicorn app.main:app --reload --port 8099
```

### 3.4 启动页面（Streamlit）

```bash
streamlit run app.py
```

打开页面后默认请求 `http://127.0.0.1:8099/api/v1`，可在左侧手动修改。

### 3.5 常见启动问题

如果你看到类似 `requesting HEAD https://huggingface.co/...` 且后端无法访问，通常是重排序模型下载阻塞。

可在 `.env` 中设置：

- `RERANKER_LOCAL_ONLY=1`（只用本地缓存，不联网下载）
- `RERANKER_ENABLED=0`（直接禁用 CrossEncoder，使用词法重排）

修改后重启 uvicorn 即可。

## 4. 接口示例

### 4.1 上传手相图并解盘

```bash
curl -X POST "http://127.0.0.1:8099/api/v1/palm/analyze" \
  -F "image=@./your_palm.jpg"
```

返回示例：

```json
{
  "session_id": "d5d7...",
  "gate": {
    "category": "palm",
    "confidence": 0.91,
    "reason": "检测到有效手掌图像。"
  },
  "profile": {
    "finger_gap": "中等",
    "life_line": "较深且连贯",
    "head_line": "平直偏长",
    "heart_line": "上扬",
    "career_line": "中段略断续",
    "sun_line": "较浅",
    "marriage_line": "浅",
    "fingerprint_pattern": "纹理可见",
    "notes": ["..."]
  },
  "report": "..."
}
```

### 4.2 基于会话继续追问

```bash
curl -X POST "http://127.0.0.1:8099/api/v1/palm/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"d5d7...","query":"我未来两年财运如何？"}'
```

## 5. 生产化建议

1. 用独立视觉模型替换当前 `VisionAgent` 降级逻辑（如 Qwen-VL / InternVL）。
2. 将知识库构建做成离线索引任务，支持增量更新。
3. 为 `trace` 增加可视化监控，记录改写和召回命中率。
4. 增加安全策略：敏感话题拦截、免责声明强化、回答长度和风险控制。

## 6. 注意事项

- 手相内容属于民俗娱乐，不应替代医学、法律、投资决策。
- 当前项目强调“工程骨架 + 可扩展性”，你可在现有接口层继续接入真实业务前端。

## 7. 知识库放置建议

推荐放置位置与用途：

- `knowledge/palm_rules.txt`: 结构化规则（线纹特征 -> 解释倾向 -> 建议动作）
- `knowledge/qa_cases.txt`: 问答案例（口语问题 -> 稳定回答模板）
- `knowledge/graph_edges.csv`: 特征关联图谱（source,target,relation,weight）
- `knowledge/*.txt`: 其他规则补充文件（会自动纳入规则检索）
- `手相学.txt`: 根目录历史知识文件（已兼容读取）

说明：

- `qa_cases.txt` 仅作为 QA 语料读取，不建议混入规则长文。
- 大模型联网时，RAG 主要用于“约束风格、补充本地业务语义、降低幻觉”，不是替代通识推理。
