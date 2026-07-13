#!/bin/bash

# Bash script to execute commands from cmds.txt sequentially
# Usage: ./run_optimizations.sh

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if cmds.txt exists
if [ ! -f "cmds.txt" ]; then
    echo -e "${RED}Error: cmds.txt not found in current directory${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Count total commands
total_commands=$(grep -c "^pnpm" cmds.txt)
current=0

echo -e "${GREEN}Starting optimization runs...${NC}"
echo -e "${GREEN}Total commands to execute: $total_commands${NC}"
echo "================================"
echo ""

# Read and execute each command from cmds.txt
while IFS= read -r cmd; do
    # Skip empty lines and comments
    if [[ -z "$cmd" ]] || [[ "$cmd" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    current=$((current + 1))
    echo -e "${YELLOW}[$current/$total_commands] Executing:${NC} $cmd"
    echo "----------------------------------------"
    
    # Execute the command
    if eval "$cmd"; then
        echo -e "${GREEN}[$current/$total_commands] ✓ Command completed successfully${NC}"
    else
        echo -e "${RED}[$current/$total_commands] ✗ Command failed with exit code $?${NC}"
        echo -e "${RED}Stopping execution...${NC}"
        exit 1
    fi
    echo ""
    
done < "cmds.txt"

echo "================================"
echo -e "${GREEN}All optimization runs completed successfully!${NC}"