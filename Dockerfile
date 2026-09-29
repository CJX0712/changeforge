# ChangeForge —— 变化点检测工程框架
# 作者: 晨星
#
# 构建: docker build -t changeforge:0.1.0 .
# 运行: docker run --rm changeforge:0.1.0 info
#       docker run --rm -v "$PWD/artifacts:/app/artifacts" changeforge:0.1.0 benchmark

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖先于源码拷贝，充分利用镜像层缓存
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY changeforge ./changeforge
COPY examples ./examples
COPY tests ./tests

# 以可编辑方式安装，使 `changeforge` 命令与包导入均可用
RUN pip install --no-deps -e .

ENV ENV_CHANGEFORGE_OUTPUT_DIR=/app/artifacts
RUN mkdir -p /app/artifacts

# 冒烟自检：确保镜像内 CLI 可用
RUN changeforge list-datasets

ENTRYPOINT ["changeforge"]
CMD ["info"]
