# Demo image for the web explorer: CPU-only PyTorch, shipped checkpoints, KITTI label files only (no images).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user cftraj ./cftraj
COPY --chown=user web ./web
COPY --chown=user checkpoints ./checkpoints
COPY --chown=user deploy/label_02 ./deploy/label_02

EXPOSE 7860
# Hosts like Render inject $PORT; default to 7860 elsewhere.
CMD ["sh", "-c", "python -c \"from cftraj.server import serve; serve('0.0.0.0', int(__import__('os').environ.get('PORT', 7860)), label_dir='deploy/label_02')\""]
