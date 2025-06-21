#!/bin/bash
# deploy.sh - Quick deployment script

set -e

echo "🚀 Deploying IPR Predictor..."

# Generate secret key if .env doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "SECRET_KEY=$(openssl rand -hex 32)" > .env
    echo "FLASK_ENV=production" >> .env
fi

# Stop existing container
echo "🛑 Stopping existing container..."
docker stop ipr-predictor 2>/dev/null || true
docker rm ipr-predictor 2>/dev/null || true

# Build new image
echo "🔨 Building Docker image..."
docker build -t ipr-predictor:latest .

# Run new container
echo "🚀 Starting new container..."
docker run -d \
  --name ipr-predictor \
  --restart unless-stopped \
  -p 80:5000 \
  --env-file .env \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models:rw \
  ipr-predictor:latest

# Wait for startup
echo "⏳ Waiting for startup..."
sleep 10

# Test health check
echo "🏥 Testing health check..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Deployment successful!"
    echo "🔗 Your app is live at: http://$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo 'YOUR_EC2_IP')"
else
    echo "❌ Health check failed!"
    docker logs ipr-predictor
    exit 1
fi