#!/bin/bash

# =============================================================================
# Video QA Validation Script for Captive Devices
# =============================================================================
#
# Usage:
#   ./validate.sh                           # Process all takes in ./data/
#   ./validate.sh /path/to/data             # Process all takes in directory
#   ./validate.sh --take /path/to/Take_007  # Process single take
#   ./validate.sh --list                    # List all discovered takes
#
# =============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/venv"
DATA_DIR="${SCRIPT_DIR}/data"

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

print_header() {
    echo ""
    echo "=============================================="
    echo "  Captive Devices - Video QA Validator"
    echo "=============================================="
    echo ""
}

check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment not found at ${VENV_PATH}${NC}"
        echo "Run setup first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

activate_venv() {
    source "${VENV_PATH}/bin/activate"
}

show_help() {
    echo "Usage: $0 [OPTIONS] [DATA_DIRECTORY]"
    echo ""
    echo "Options:"
    echo "  --take PATH    Process a single take folder"
    echo "  --list         List all discovered takes without processing"
    echo "  --debug        Save debug frames for flagged issues"
    echo "  --verbose      Enable verbose logging"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Process all takes in ./data/"
    echo "  $0 /Volumes/Shoots/Project1     # Process all takes in directory"
    echo "  $0 --take ./data/pull3/Take_007 # Process single take"
    echo "  $0 --list                       # List discovered takes"
    echo ""
}

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------

SINGLE_TAKE=""
LIST_ONLY=""
DEBUG_FLAG=""
VERBOSE_FLAG=""
CUSTOM_DATA_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --take|-t)
            SINGLE_TAKE="$2"
            shift 2
            ;;
        --list|-l)
            LIST_ONLY="--list-only"
            shift
            ;;
        --debug|-d)
            DEBUG_FLAG="--debug"
            shift
            ;;
        --verbose|-v)
            VERBOSE_FLAG="--verbose"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            CUSTOM_DATA_DIR="$1"
            shift
            ;;
    esac
done

# Use custom data dir if provided
if [ -n "$CUSTOM_DATA_DIR" ]; then
    DATA_DIR="$CUSTOM_DATA_DIR"
fi

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

print_header

# Check and activate virtual environment
check_venv
activate_venv

echo -e "${GREEN}Virtual environment activated${NC}"
echo ""

# Run appropriate command
if [ -n "$SINGLE_TAKE" ]; then
    # Single take mode
    echo "Processing single take: ${SINGLE_TAKE}"
    echo ""
    python "${SCRIPT_DIR}/batch_validate.py" --take "$SINGLE_TAKE" $DEBUG_FLAG $VERBOSE_FLAG
    
elif [ -n "$LIST_ONLY" ]; then
    # List mode
    echo "Discovering takes in: ${DATA_DIR}"
    python "${SCRIPT_DIR}/batch_validate.py" "$DATA_DIR" --list-only
    
else
    # Batch mode
    echo "Processing all takes in: ${DATA_DIR}"
    echo ""
    python "${SCRIPT_DIR}/batch_validate.py" "$DATA_DIR" $DEBUG_FLAG $VERBOSE_FLAG
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Validation completed successfully${NC}"
else
    echo -e "${RED}Validation completed with issues${NC}"
fi

exit $EXIT_CODE
