# AI System - Conceptual Docker Configuration
#
# Demonstrates containerization patterns for AI applications
# Abstract multi-stage build concept

# Build Stage - Concept: Dependency Preparation
FROM python:3.11-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt
COPY src/ ./src/

# Production Stage - Concept: Runtime Environment

# Concept: Security patterns

# Concept: Dependency isolation

# Concept: Environment setup
WORKDIR /app
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH

# Concept: Service configuration
ENV LOG_LEVEL=INFO
EXPOSE 8000

# Concept: Health monitoring
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1

# Concept: Application startup
CMD ["python", "-m", "src.main"]

# Development Stage - Concept: Development Tools

# GPU Stage - Concept: Accelerated Computing
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu-enabled
RUN apt-get update && apt-get install -y python3 python3-pip
COPY --from=builder /build/src /app/src
WORKDIR /app
ENV CUDA_VISIBLE_DEVICES=0
CMD ["python3", "-m", "src.training"] 