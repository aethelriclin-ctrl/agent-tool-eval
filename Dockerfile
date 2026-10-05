# Agent 工具调用评测 —— 最小可复现环境
#
# 用法：
#   1) 先建一个只含 key 的 env 文件（不要提交到仓库）：
#        echo DEEPSEEK_API_KEY=sk-你的key > .env
#   2) 构建：
#        docker build -t agent-eval .
#   3) 跑（默认跑「手写版」8 题）：
#        docker run --rm --env-file .env agent-eval
#   4) 跑「框架版」（LangGraph）：
#        docker run --rm --env-file .env agent-eval python agent_eval.py --impl=lg
#
# ⚠️ 状态：Dockerfile 已写，但**尚未在本地成功构建验证**（本机 WSL/Docker 环境未就绪）。
#         首次构建时请以实际报错为准调整。

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先只拷依赖清单，让依赖层可缓存（改代码不会重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码与题库
COPY agent_eval.py agent_cases.json ./
COPY langgraph_impl/ ./langgraph_impl/

# 默认跑手写版（依赖最小）；框架版见文件顶部用法第 4 条
CMD ["python", "agent_eval.py"]
