#!/bin/bash

# ===================================
# DDSP-SVC-Enhanced Quick Start Script
# ===================================

set -e  # Exit on error

echo "🚀 Starting DDSP-SVC-Enhanced..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate || source .venv/Scripts/activate 2>/dev/null
elif [ -d "venv" ]; then
    source venv/bin/activate || source venv/Scripts/activate 2>/dev/null
fi

echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Start API backend
echo "📡 Starting API backend..."
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!
cd ..

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 3

# Start frontend
echo "🌐 Starting web frontend..."
cd web
npm run dev &
WEB_PID=$!
cd ..

echo ""
echo -e "${GREEN}✅ DDSP-SVC-Enhanced is running!${NC}"
echo ""
echo "📡 API Backend:  http://localhost:8000"
echo "📚 API Docs:     http://localhost:8000/docs"
echo "🌐 Web Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"

# Trap Ctrl+C and cleanup
trap "echo ''; echo 'Stopping services...'; kill $API_PID $WEB_PID 2>/dev/null; exit" INT

# Wait for processes
wait
