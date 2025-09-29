#!/bin/bash

echo "🚀 Starting Resume Analyzer Backend with Docker"
echo "================================================"

# Build and run the container
docker-compose up --build -d

echo "✅ Backend is running at http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo ""
echo "🔧 Commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop: docker-compose down"
echo "   • Restart: docker-compose restart"
