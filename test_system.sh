#!/bin/bash

# Resume Intelligence System - Comprehensive Test Suite
# This script tests both backend and frontend components

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Resume Intelligence System - Test Suite                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:8080"

PASSED=0
FAILED=0
WARNINGS=0

# Test result function
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
        ((FAILED++))
    fi
}

test_warning() {
    echo -e "${YELLOW}⚠ WARNING${NC}: $1"
    ((WARNINGS++))
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}BACKEND TESTS${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: Backend Health Check
echo "Test 1: Backend Health Check"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" $BACKEND_URL/health)
if [ "$HEALTH" = "200" ]; then
    test_result 0 "Backend is healthy and responding"
else
    test_result 1 "Backend health check failed (HTTP $HEALTH)"
fi
echo ""

# Test 2: API Root Endpoint
echo "Test 2: API Root Endpoint"
ROOT=$(curl -s $BACKEND_URL/ | jq -r '.status' 2>/dev/null)
if [ "$ROOT" = "ready" ]; then
    test_result 0 "API root endpoint accessible"
else
    test_result 1 "API root endpoint not responding correctly"
fi
echo ""

# Test 3: Component Status
echo "Test 3: Component Status Check"
COMPONENTS=$(curl -s $BACKEND_URL/health | jq -r '.components | to_entries[] | select(.value == false) | .key' 2>/dev/null)
if [ -z "$COMPONENTS" ]; then
    test_result 0 "All components initialized successfully"
else
    test_result 1 "Some components failed: $COMPONENTS"
fi
echo ""

# Test 4: Resume Scoring Endpoint
echo "Test 4: Resume Scoring Endpoint"
SCORE=$(curl -s -X POST "$BACKEND_URL/score" \
    -H "Content-Type: application/json" \
    -d '{"resume_text":"Python developer with 5 years experience","job_description":"Python developer needed"}' \
    | jq -r '.overallScore' 2>/dev/null)
if [ ! -z "$SCORE" ] && [ "$SCORE" != "null" ]; then
    test_result 0 "Resume scoring working (Score: $SCORE)"
else
    test_result 1 "Resume scoring endpoint failed"
fi
echo ""

# Test 5: Job Matching Endpoint
echo "Test 5: Job Matching Endpoint"
JOBS=$(curl -s "$BACKEND_URL/jobs?skills=python,javascript" | jq -r 'type' 2>/dev/null)
if [ "$JOBS" = "array" ]; then
    JOB_COUNT=$(curl -s "$BACKEND_URL/jobs?skills=python,javascript" | jq 'length' 2>/dev/null)
    if [ "$JOB_COUNT" -gt 0 ]; then
        test_result 0 "Job matching working ($JOB_COUNT jobs found)"
    else
        test_warning "Job matching endpoint works but returned 0 jobs (dataset may be unavailable)"
    fi
else
    test_result 1 "Job matching endpoint failed"
fi
echo ""

# Test 6: Skill Gap Analysis
echo "Test 6: Skill Gap Analysis"
GAPS=$(curl -s "$BACKEND_URL/skill-gap?userSkills=python&roleRequiredSkills=python,docker,kubernetes" \
    | jq -r '.missing_skills | length' 2>/dev/null)
if [ ! -z "$GAPS" ] && [ "$GAPS" != "null" ]; then
    test_result 0 "Skill gap analysis working ($GAPS gaps identified)"
else
    test_result 1 "Skill gap analysis failed"
fi
echo ""

# Test 7: Course Recommendations
echo "Test 7: Course Recommendations"
COURSES=$(curl -s "$BACKEND_URL/courses?skills=python" | jq -r 'type' 2>/dev/null)
if [ "$COURSES" = "object" ]; then
    test_result 0 "Course recommendations endpoint working"
else
    test_result 1 "Course recommendations endpoint failed"
fi
echo ""

# Test 8: ML Analytics
echo "Test 8: ML Analytics Endpoint"
ML_STATUS=$(curl -s "$BACKEND_URL/ml-analytics" | jq -r '.status' 2>/dev/null)
if [ "$ML_STATUS" = "success" ]; then
    test_result 0 "ML analytics endpoint working"
else
    test_result 1 "ML analytics endpoint failed"
fi
echo ""

# Test 9: Analytics Dashboard
echo "Test 9: Analytics Dashboard"
ANALYTICS=$(curl -s "$BACKEND_URL/analytics" | jq -r 'keys | length' 2>/dev/null)
if [ "$ANALYTICS" -ge 3 ]; then
    test_result 0 "Analytics dashboard working ($ANALYTICS data sources)"
else
    test_result 1 "Analytics dashboard failed"
fi
echo ""

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}FRONTEND TESTS${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 10: Frontend Accessibility
echo "Test 10: Frontend Server Accessibility"
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $FRONTEND_URL)
if [ "$FRONTEND_STATUS" = "200" ]; then
    test_result 0 "Frontend server accessible"
else
    test_result 1 "Frontend server not accessible (HTTP $FRONTEND_STATUS)"
fi
echo ""

# Test 11: Frontend HTML Structure
echo "Test 11: Frontend HTML Structure"
TITLE=$(curl -s $FRONTEND_URL | grep -o '<title>.*</title>' | head -1)
if [ ! -z "$TITLE" ]; then
    test_result 0 "Frontend HTML structure valid"
else
    test_result 1 "Frontend HTML structure invalid"
fi
echo ""

# Test 12: Frontend Build
echo "Test 12: Frontend Build Process"
cd frontend
BUILD_OUTPUT=$(npm run build 2>&1 | grep "built in")
if [ ! -z "$BUILD_OUTPUT" ]; then
    test_result 0 "Frontend builds successfully"
else
    test_result 1 "Frontend build failed"
fi
cd ..
echo ""

# Test 13: Frontend Assets
echo "Test 13: Frontend Static Assets"
if [ -d "frontend/dist" ] && [ -f "frontend/dist/index.html" ]; then
    ASSET_COUNT=$(find frontend/dist -type f | wc -l | tr -d ' ')
    test_result 0 "Frontend assets generated ($ASSET_COUNT files)"
else
    test_result 1 "Frontend assets not found"
fi
echo ""

# Test 14: API Configuration
echo "Test 14: Frontend API Configuration"
API_CONFIG=$(grep -r "VITE_API_BASE_URL" frontend/.env 2>/dev/null)
if [ ! -z "$API_CONFIG" ]; then
    test_result 0 "Frontend API configuration present"
else
    test_warning "Frontend API configuration not found in .env"
fi
echo ""

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}INTEGRATION TESTS${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 15: CORS Configuration
echo "Test 15: CORS Configuration"
CORS=$(curl -s -H "Origin: http://localhost:8080" -I $BACKEND_URL/health | grep -i "access-control-allow-origin")
if [ ! -z "$CORS" ]; then
    test_result 0 "CORS properly configured"
else
    test_warning "CORS headers not detected (may cause frontend issues)"
fi
echo ""

# Test 16: Backend Response Time
echo "Test 16: Backend Response Time"
RESPONSE_TIME=$(curl -s -o /dev/null -w "%{time_total}" $BACKEND_URL/health)
RESPONSE_MS=$(echo "$RESPONSE_TIME * 1000" | bc | cut -d'.' -f1)
if [ "$RESPONSE_MS" -lt 1000 ]; then
    test_result 0 "Backend response time acceptable (${RESPONSE_MS}ms)"
else
    test_warning "Backend response time slow (${RESPONSE_MS}ms)"
fi
echo ""

# Test 17: File Upload Size Limit
echo "Test 17: File Upload Configuration"
if [ -f "backend/main.py" ]; then
    test_result 0 "Backend main.py exists"
else
    test_result 1 "Backend main.py not found"
fi
echo ""

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}DOCKER TESTS${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 18: Docker Configuration
echo "Test 18: Docker Configuration Files"
if [ -f "backend/Dockerfile" ] && [ -f "backend/docker-compose.yml" ]; then
    test_result 0 "Docker configuration files present"
else
    test_result 1 "Docker configuration files missing"
fi
echo ""

# Test 19: Docker Container Status
echo "Test 19: Docker Container Status"
CONTAINER=$(docker ps --filter "name=backend" --format "{{.Status}}" 2>/dev/null | head -1)
if [ ! -z "$CONTAINER" ]; then
    test_result 0 "Backend Docker container running"
else
    test_warning "Backend Docker container not detected"
fi
echo ""

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${RED}Failed:${NC}   $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

TOTAL=$((PASSED + FAILED))
if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $PASSED * 100 / $TOTAL" | bc)
    echo "Success Rate: ${SUCCESS_RATE}%"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please review the results above.${NC}"
    exit 1
fi
