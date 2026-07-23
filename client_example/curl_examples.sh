#!/bin/bash
# ===================================================================
# Face Clustering API - cURL Test Examples
# 
# Prerequisites:
#   - API server running on localhost:8000
#   - curl and jq installed
#   - Sample images in ./sample_images/
# ===================================================================

API_URL="http://localhost:8000"
SAMPLE_DIR="./sample_images"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}  Face Clustering API - cURL Tests      ${NC}"
echo -e "${YELLOW}=========================================${NC}"

# -------------------------------------------------------------------
# 1. Health Check
echo -e "\n${YELLOW}[1/5] Health Check${NC}"
curl -s "${API_URL}/health" | jq .

# -------------------------------------------------------------------
# 2. Service Info
echo -e "\n${YELLOW}[2/5] Service Info${NC}"
curl -s "${API_URL}/info" | jq .

# -------------------------------------------------------------------
# 3. Cluster Single Image
echo -e "\n${YELLOW}[3/5] Cluster Single Image${NC}"
if [ -f "${SAMPLE_DIR}/person1.jpg" ]; then
    curl -s -X POST "${API_URL}/cluster" \
        -F "images=@${SAMPLE_DIR}/person1.jpg" \
        -F "eps=0.4" \
        -F "min_samples=2" \
        -F "detection_model=hog" | jq .
else
    echo -e "${RED}  Skipped: ${SAMPLE_DIR}/person1.jpg not found${NC}"
    echo "  Place test images in ${SAMPLE_DIR}/ to run this test"
fi

# -------------------------------------------------------------------
# 4. Cluster Multiple Images
echo -e "\n${YELLOW}[4/5] Cluster Multiple Images${NC}"
if [ -d "${SAMPLE_DIR}" ] && [ "$(ls -A ${SAMPLE_DIR}/*.jpg 2>/dev/null)" ]; then
    # Build multipart form with all images
    FORM_ARGS=()
    for img in ${SAMPLE_DIR}/*.jpg ${SAMPLE_DIR}/*.png; do
        [ -f "$img" ] && FORM_ARGS+=(-F "images=@${img}")
    done
    
    if [ ${#FORM_ARGS[@]} -gt 0 ]; then
        curl -s -X POST "${API_URL}/cluster" \
            "${FORM_ARGS[@]}" \
            -F "eps=0.4" \
            -F "min_samples=2" | jq .
    else
        echo -e "${RED}  No images found in ${SAMPLE_DIR}${NC}"
    fi
else
    echo -e "${RED}  Skipped: ${SAMPLE_DIR} empty or not found${NC}"
fi

# -------------------------------------------------------------------
# 5. Error Handling Test - No images
echo -e "\n${YELLOW}[5/5] Error Handling - Empty Request${NC}"
curl -s -X POST "${API_URL}/cluster" | jq .

echo -e "\n${GREEN}All tests completed!${NC}"
