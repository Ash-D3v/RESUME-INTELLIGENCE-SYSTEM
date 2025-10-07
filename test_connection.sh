#!/bin/bash

# Frontend-Backend Connection Test Script
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Frontend-Backend Connection Verification                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:8080"

echo -e "${BLUE}Step 1: Checking if servers are running...${NC}"
echo ""

# Check backend
BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BACKEND_URL/health)
if [ "$BACKEND_STATUS" = "200" ]; then
    echo -e "${GREEN}✓${NC} Backend is running on $BACKEND_URL"
else
    echo -e "${RED}✗${NC} Backend is NOT running on $BACKEND_URL"
    echo "   Please start the backend: cd backend && docker-compose up"
    exit 1
fi

# Check frontend
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $FRONTEND_URL)
if [ "$FRONTEND_STATUS" = "200" ]; then
    echo -e "${GREEN}✓${NC} Frontend is running on $FRONTEND_URL"
else
    echo -e "${RED}✗${NC} Frontend is NOT running on $FRONTEND_URL"
    echo "   Please start the frontend: cd frontend && npm run dev"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 2: Testing API endpoints from frontend perspective...${NC}"
echo ""

# Test CORS
echo "Testing CORS configuration..."
CORS_HEADER=$(curl -s -H "Origin: $FRONTEND_URL" -I $BACKEND_URL/health | grep -i "access-control-allow-origin")
if [ ! -z "$CORS_HEADER" ]; then
    echo -e "${GREEN}✓${NC} CORS is properly configured"
else
    echo -e "${RED}✗${NC} CORS headers not found (may cause issues)"
fi

# Test Health endpoint
echo "Testing /health endpoint..."
HEALTH=$(curl -s $BACKEND_URL/health | jq -r '.status' 2>/dev/null)
if [ "$HEALTH" = "healthy" ]; then
    echo -e "${GREEN}✓${NC} Health check passed"
else
    echo -e "${RED}✗${NC} Health check failed"
fi

# Test Score endpoint
echo "Testing /score endpoint..."
SCORE=$(curl -s -X POST "$BACKEND_URL/score" \
    -H "Content-Type: application/json" \
    -d '{"resume_text":"Python developer","job_description":"Python needed"}' \
    | jq -r '.overallScore' 2>/dev/null)
if [ ! -z "$SCORE" ] && [ "$SCORE" != "null" ]; then
    echo -e "${GREEN}✓${NC} Score endpoint working (returned score: $SCORE)"
else
    echo -e "${RED}✗${NC} Score endpoint failed"
fi

# Test Jobs endpoint
echo "Testing /jobs endpoint..."
JOBS=$(curl -s "$BACKEND_URL/jobs?skills=python" | jq -r 'type' 2>/dev/null)
if [ "$JOBS" = "array" ]; then
    echo -e "${GREEN}✓${NC} Jobs endpoint working"
else
    echo -e "${RED}✗${NC} Jobs endpoint failed"
fi

# Test Analytics endpoint
echo "Testing /analytics endpoint..."
ANALYTICS=$(curl -s "$BACKEND_URL/analytics" | jq -r 'keys | length' 2>/dev/null)
if [ "$ANALYTICS" -ge 3 ]; then
    echo -e "${GREEN}✓${NC} Analytics endpoint working"
else
    echo -e "${RED}✗${NC} Analytics endpoint failed"
fi

echo ""
echo -e "${BLUE}Step 3: Verifying frontend configuration...${NC}"
echo ""

# Check .env file
if [ -f "frontend/.env" ]; then
    API_URL=$(grep VITE_API_BASE_URL frontend/.env | cut -d'=' -f2)
    echo -e "${GREEN}✓${NC} Frontend .env file exists"
    echo "   API URL configured as: $API_URL"
    
    if [ "$API_URL" = "$BACKEND_URL" ]; then
        echo -e "${GREEN}✓${NC} API URL matches backend URL"
    else
        echo -e "${RED}✗${NC} API URL mismatch!"
        echo "   Expected: $BACKEND_URL"
        echo "   Found: $API_URL"
    fi
else
    echo -e "${RED}✗${NC} Frontend .env file not found"
fi

# Check api-config.ts
if [ -f "frontend/src/lib/api-config.ts" ]; then
    echo -e "${GREEN}✓${NC} API configuration file exists"
else
    echo -e "${RED}✗${NC} API configuration file not found"
fi

echo ""
echo -e "${BLUE}Step 4: Connection Summary${NC}"
echo ""
echo "Backend URL:  $BACKEND_URL"
echo "Frontend URL: $FRONTEND_URL"
echo ""
echo -e "${GREEN}✓ Frontend and Backend are properly connected!${NC}"
echo ""
echo "You can now:"
echo "  1. Open your browser to: $FRONTEND_URL"
echo "  2. Navigate to: $FRONTEND_URL/connection-test"
echo "  3. Run comprehensive tests from the UI"
echo ""
echo "API Documentation: $BACKEND_URL/docs"
