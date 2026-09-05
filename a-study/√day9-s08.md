# 第9天：s08 Context Compact

今天只解决一个核心问题：

> Agent 的消息历史不断增长时，Harness 如何按成本和信息损失从低到高逐步腾出上下文空间？

建议投入 **90分钟**。

今天不追求把整个`s08`迁移到OpenAI。先把四层压缩的定位、顺序和前三层确定性行为真正看懂并验证；OpenAI消息结构适配与降低阈值观察自动摘要放到第10天。

## 今日定位

- 所属层：Harness上下文管理层
- 解决问题：长任务中的文件内容、命令输出和对话历史最终超过模型上下文窗口
- 接入位置：每次调用模型之前
- 重要程度：核心
- 今日深度：理解完整管线，实际验证前三层，不深挖模型摘要实现

先建立总图：

```text
messages持续增长
       ↓
1. tool_result_budget：超大工具结果落盘
       ↓
2. snip_compact：消息过多时归档中间历史
       ↓
3. micro_compact：已被模型看过的旧工具结果缩成占位符
       ↓
4. compact_history：仍超限时调用模型总结历史
       ↓
携带压缩后的messages继续执行
```

这四步的排序原则：

```text
低成本、可恢复、信息损失小
→ 高成本、不可完全恢复、可能遗漏细节
```

## 今日可见成果

今天结束时必须亲眼看到：

1. 大工具结果被写入`.task_outputs/tool-results/`
2. 上下文只保留文件路径和内容预览
3. 消息数量超限后，中间历史写入`.transcripts/`
4. 较早的已消费工具结果变成占位符
5. 最近的工具结果仍然保留
6. 能解释为什么模型摘要必须最后执行

## 本地代码现状

当前`s08_context_compact/code.py`：

- 仍使用Anthropic API
- 消息判断依赖`tool_use`和`tool_result`
- 包含四层压缩和一次API超长后的补救压缩
- 自动摘要会真实调用模型
- 前三层只是本地文件与数据结构操作，不需要模型

因此今天不要直接做完整OpenAI迁移。OpenAI Responses使用`function_call`和`function_call_output`，如果直接替换API而不重写消息识别函数，压缩器会悄悄失效。

## 0～10分钟：先理解上下文是什么

阅读`s08_context_compact/README.zh.md`中的：

- 先理解上下文
- 为什么先整理工具结果
- 为什么顺序固定
- 与`s09`的边界

回答：

```text
上下文是模型本轮请求能够看到的全部输入。
messages是Harness维护上下文的一种数据结构。
上下文窗口是模型能接收的容量上限。
```

不要把三个概念混为一谈。

### s08与s09的边界

```text
s08 Context Compact
→ 管理当前会话里有限的上下文
→ 允许裁剪可恢复的细节

s09 Memory
→ 保存跨压缩、跨会话仍需存在的信息
→ 关注什么值得长期记住
```

## 10～25分钟：定位四层压缩

打开`s08_context_compact/code.py`，不要从第一行逐行读，只定位：

```python
class ContextCompactor

tool_result_budget()
snip_compact()
micro_compact()
compact_history()
reactive_compact()
prepare()
```

重点看`prepare()`：

```python
def prepare(self, messages, active_request):
    messages = self.tool_result_budget(messages)
    messages = self.snip_compact(messages)
    messages = self.micro_compact(messages)
    if self.estimate_chars(messages) > self.CONTEXT_CHAR_LIMIT:
        messages = self.compact_history(messages, active_request)
    return messages
```

今天需要理解的是顺序，而不是背实现。

填写这张表：

| 层 | 触发依据 | 处理对象 | 是否调用模型 | 是否可恢复 |
|---|---|---|---|---|
| tool_result_budget | 最新批次工具结果字符数 | 超大工具输出 | 否 | 是，已落盘 |
| snip_compact | 消息数量 | 中间历史 | 否 | 是，已归档 |
| micro_compact | 旧且已消费的工具结果 | 工具结果正文 | 否 | 部分可恢复 |
| compact_history | 总字符数 | 整段历史 | 是 | 原文已归档，但摘要有损 |

## 25～35分钟：理解“已看过”和“未看过”

重点阅读：

```python
unseen_tool_result_positions()
micro_compact()
```

关键规则：

> 刚执行完、还没有交给模型读取的工具结果不能被提前压缩。

数据流：

```text
模型返回tool_use
→ Harness执行工具
→ 新增tool_result
→ 下一轮模型尚未读取
→ 该结果属于unseen，必须完整保留

下一次模型响应已经生成
→ 说明模型读过此前tool_result
→ 它变成consumed，可以参与旧结果裁剪
```

如果把未读取结果提前替换成占位符，模型从未真正看到工具输出，可能错误判断任务已经完成。

## 35～55分钟：建立离线压缩实验

让Codex创建一个独立实验文件：

```text
请创建a-study/day9_context_compact_demo.py，用合成消息离线演示
s08的前三层上下文压缩，不发起任何API请求。

要求：

1. 复用s08_context_compact/code.py中的ContextCompactor。
2. 导入模块前设置假的ANTHROPIC_API_KEY和MODEL_ID，确保仅初始化SDK，
   但实验中不得调用summarize_history或任何模型API。
3. 使用临时目录保存transcript和tool result，不能污染重要目录。
4. 实验A：构造一个超过1000字符的tool_result，把
   LARGE_RESULT_CHAR_LIMIT和批次预算调低，调用tool_result_budget，
   打印压缩前后长度和落盘路径。
5. 实验B：构造12条成对的assistant(tool_use)和user(tool_result)，
   用max_messages=6调用snip_compact，打印压缩前后消息数和归档路径。
6. 实验C：构造5个已经被模型消费的长tool_result，以及1个尚未
   被模型消费的新结果，调用micro_compact，打印每个结果压缩后的内容。
7. 必须证明最新未消费结果保持完整。
8. 只创建实验文件，不修改s08原始代码。
9. 所有注释和输出使用中文。
10. 完成后运行py_compile和实验脚本。
```

你的所有权动作不是手写代码，而是先预测三个实验的结果，再审查Codex是否真的没有调用API。

## 55～70分钟：运行并观察三个实验

### 实验A：大结果落盘

预期：

```text
压缩前：完整长文本
压缩后：
<persisted-output>
Full output: 临时目录中的文件路径
Preview: 前一部分文本
</persisted-output>
```

检查磁盘文件仍包含完整原文。

### 实验B：中间历史归档

预期：

```text
最早少量消息保留
中间消息被归档标记替代
最近消息保留
.transcripts或临时transcript目录出现完整历史
```

重点检查切点没有拆散：

```text
assistant(tool_use)
user(tool_result)
```

### 实验C：旧结果占位

预期：

```text
早期已消费长结果 → [Earlier tool result omitted.]
最近3条已消费结果 → 保留
最新未消费结果 → 完整保留
```

如果未消费结果被裁剪，说明合成消息顺序或判断逻辑有问题，不要继续。

## 70～80分钟：理解为什么不能随便删除消息

今天必须理解两类不能拆散的关系。

### 工具调用配对

```text
tool_use(id=123)
↔ tool_result(tool_use_id=123)
```

删除一边可能导致API认为消息历史非法。

迁移到OpenAI后对应：

```text
function_call(call_id=123)
↔ function_call_output(call_id=123)
```

### 当前用户请求

工具结果在Anthropic格式中也使用`role=user`，所以不能简单把“最后一个user消息”当作用户原始请求。

本地代码把`active_request`单独传入：

```python
agent_loop(history, query)
```

压缩历史时再明确写入：

```text
Current user request
Conversation summary
```

这样摘要不会覆盖用户当前目标。

## 80～85分钟：理解模型摘要层

今天只追踪，不运行：

```python
summarize_history()
summary_message()
compact_history()
```

模型摘要的优点：

- 压缩比例高
- 能保留目标、决定和剩余任务

模型摘要的代价：

- 额外API调用和费用
- 摘要可能遗漏细节
- 历史中的恶意指令可能污染摘要
- 摘要质量依赖模型

所以它必须排在确定性整理之后。

## 85～90分钟：闭卷复述

填写：

```text
s08解决的问题：
压缩发生在模型调用的什么位置：
四层压缩顺序：
第一层为什么先落盘：
未消费工具结果为什么不能压缩：
snip时为什么保护工具调用配对：
模型摘要为什么最后执行：
Context Compact与Memory的区别：
```

## 数据流

```text
当前messages
→ 大结果落盘并保留预览
→ 中间历史归档
→ 旧工具结果缩成占位符
→ 估算剩余字符数
→ 未超限：调用业务模型
→ 仍超限：总结历史后调用业务模型
→ API仍报超长：补救压缩并重试一次
```

## 验收标准

- 能按顺序说出四层压缩及其成本差异
- 实际看到大结果落盘且原文可恢复
- 实际看到中间历史归档且调用/结果没有被拆散
- 实际看到旧结果被替换、未消费结果保持完整
- 能解释`s08 Context`与`s09 Memory`的边界
- 实验全程没有发起模型API请求

## 今天不要做

- 不完整迁移`s08`到OpenAI
- 不降低自动摘要阈值并调用模型，那是第10天
- 不研究精确Token计算
- 不实现向量数据库或长期记忆
- 不优化摘要Prompt
- 不把所有历史直接删除
- 不把上下文压缩误认为长期记忆
- 不修改`s08`原始代码
- 不进入`s09`

今天最重要的认知：

> 上下文压缩不是简单地“删掉旧消息”，而是按可恢复性和信息价值分层处理：先落盘、再裁剪、后总结，并始终保护当前目标和工具调用配对。
