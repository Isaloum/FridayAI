#!/bin/bash

# =====================================
# FridayAI Repository Cleanup Script
# =====================================
# This script removes sensitive files and large binaries from git history
# using BFG Repo-Cleaner (preferred) or git filter-branch (fallback)
#
# WARNING: This script rewrites git history and requires force-push
# Make sure all team members are aware before running
#
# Usage:
#   ./cleanup.sh --dry-run    # Preview changes without applying
#   ./cleanup.sh              # Run actual cleanup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
BACKUP_DIR="${HOME}/fridayai-cleanup-backup-$(date +%Y%m%d-%H%M%S)"
BFG_VERSION="1.14.0"
BFG_JAR="bfg-${BFG_VERSION}.jar"
BFG_URL="https://repo1.maven.org/maven2/com/madgag/bfg/${BFG_VERSION}/bfg-${BFG_VERSION}.jar"

# Files and patterns to remove
SENSITIVE_FILES=(
    "*.pem"
    "*.key"
    "github-token-2025.txt"
    "friday_memory.enc"
    "test_query.enc"
    "vault.key"
    "memory.key"
    "FridayAI-New.pem"
    "FridayAIKey.pem"
    "fridayai-backend.pem"
)

LARGE_BINARIES=(
    "python-3.11.6-amd64.exe"
    "python-3.11.6-amd64 (1).exe"
    "python-3.13.5-amd64.exe"
    "rustup-init.exe"
    "swigwin-3.0.12.zip"
    "swigwin-4.3.1.zip"
)

DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--help]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Preview changes without applying them"
            echo "  --help       Show this help message"
            exit 0
            ;;
    esac
done

# =====================================
# Helper Functions
# =====================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi
    print_success "Git repository detected"
    
    # Check for Java (required for BFG)
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1)
        print_success "Java found: ${JAVA_VERSION}"
    else
        print_warning "Java not found - will use git filter-branch fallback"
        return 1
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_error "You have uncommitted changes. Please commit or stash them first."
        exit 1
    fi
    print_success "No uncommitted changes"
    
    return 0
}

create_backup() {
    print_header "Creating Backup"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN: Would create backup at ${BACKUP_DIR}"
        return
    fi
    
    print_info "Creating backup at ${BACKUP_DIR}"
    mkdir -p "${BACKUP_DIR}"
    
    # Clone the current repository to backup location
    git clone --mirror "${REPO_DIR}/.git" "${BACKUP_DIR}/git-backup.git"
    
    # Also create a bundle as additional safety
    git bundle create "${BACKUP_DIR}/repo-backup.bundle" --all
    
    print_success "Backup created successfully"
    print_info "Backup location: ${BACKUP_DIR}"
    print_warning "Keep this backup until you're sure the cleanup was successful!"
}

download_bfg() {
    print_header "Setting Up BFG Repo-Cleaner"
    
    if [ -f "${BFG_JAR}" ]; then
        print_success "BFG already downloaded"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN: Would download BFG from ${BFG_URL}"
        return
    fi
    
    print_info "Downloading BFG Repo-Cleaner..."
    if command -v curl &> /dev/null; then
        curl -L -o "${BFG_JAR}" "${BFG_URL}"
    elif command -v wget &> /dev/null; then
        wget -O "${BFG_JAR}" "${BFG_URL}"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    print_success "BFG downloaded successfully"
}

show_file_sizes() {
    print_header "Current Repository Statistics"
    
    echo "Repository size:"
    du -sh "${REPO_DIR}"
    
    echo ""
    echo "Git directory size:"
    du -sh "${REPO_DIR}/.git"
    
    echo ""
    echo "Large files to be removed:"
    for file in "${LARGE_BINARIES[@]}"; do
        if [ -f "${REPO_DIR}/${file}" ]; then
            ls -lh "${REPO_DIR}/${file}" | awk '{print "  " $9 " (" $5 ")"}'
        fi
    done
    
    echo ""
    echo "Sensitive files to be removed:"
    for pattern in "${SENSITIVE_FILES[@]}"; do
        find "${REPO_DIR}" -name "${pattern}" -type f 2>/dev/null | while read -r file; do
            basename "$file"
        done | head -5
    done
}

cleanup_with_bfg() {
    print_header "Cleaning Up with BFG Repo-Cleaner"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN: Would remove files using BFG"
        for file in "${LARGE_BINARIES[@]}" "${SENSITIVE_FILES[@]}"; do
            echo "  - ${file}"
        done
        return
    fi
    
    # Remove large binaries
    print_info "Removing large binary files..."
    for file in "${LARGE_BINARIES[@]}"; do
        print_info "  Deleting: ${file}"
        java -jar "${BFG_JAR}" --delete-files "${file}" "${REPO_DIR}"
    done
    
    # Remove sensitive files by pattern
    print_info "Removing sensitive files..."
    for pattern in "${SENSITIVE_FILES[@]}"; do
        print_info "  Deleting: ${pattern}"
        java -jar "${BFG_JAR}" --delete-files "${pattern}" "${REPO_DIR}"
    done
    
    print_success "BFG cleanup completed"
}

cleanup_with_filter_branch() {
    print_header "Cleaning Up with Git Filter-Branch"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN: Would remove files using git filter-branch"
        for file in "${LARGE_BINARIES[@]}" "${SENSITIVE_FILES[@]}"; do
            echo "  - ${file}"
        done
        return
    fi
    
    print_warning "Using git filter-branch (slower than BFG)"
    
    # Build the file list
    ALL_FILES=("${LARGE_BINARIES[@]}" "${SENSITIVE_FILES[@]}")
    
    # Remove files from history
    for file in "${ALL_FILES[@]}"; do
        print_info "Removing: ${file}"
        git filter-branch --force --index-filter \
            "git rm --cached --ignore-unmatch '${file}' '**/${file}'" \
            --prune-empty --tag-name-filter cat -- --all
    done
    
    print_success "Filter-branch cleanup completed"
}

cleanup_refs_and_gc() {
    print_header "Cleaning Up References and Running Garbage Collection"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN: Would clean up refs and run git gc"
        return
    fi
    
    # Clean up the mess left by filter-branch
    print_info "Removing backup refs..."
    rm -rf .git/refs/original/
    
    print_info "Expiring reflog..."
    git reflog expire --expire=now --all
    
    print_info "Running garbage collection (this may take a while)..."
    git gc --prune=now --aggressive
    
    print_success "Repository cleaned and optimized"
}

show_results() {
    print_header "Cleanup Results"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN completed - no changes were made"
        return
    fi
    
    echo "New repository size:"
    du -sh "${REPO_DIR}"
    
    echo ""
    echo "New git directory size:"
    du -sh "${REPO_DIR}/.git"
    
    print_success "Cleanup completed successfully!"
}

show_next_steps() {
    print_header "Next Steps"
    
    if [ "$DRY_RUN" = true ]; then
        echo "To run the actual cleanup:"
        echo "  ./cleanup.sh"
        return
    fi
    
    echo "1. Verify the changes:"
    echo "   git log --all --oneline | head -20"
    echo ""
    echo "2. Check that sensitive files are gone:"
    echo "   git log --all --full-history -- '*.pem'"
    echo ""
    echo "3. If everything looks good, force push to remote:"
    echo "   git push origin --force --all"
    echo "   git push origin --force --tags"
    echo ""
    print_warning "WARNING: Force pushing will rewrite history for all collaborators!"
    print_warning "Make sure everyone has committed their work first."
    echo ""
    echo "4. All team members must re-clone the repository:"
    echo "   cd .."
    echo "   rm -rf FridayAI"
    echo "   git clone https://github.com/Isaloum/FridayAI.git"
    echo ""
    echo "5. If something goes wrong, restore from backup:"
    echo "   See CLEANUP_INSTRUCTIONS.md for rollback procedure"
    echo "   Backup location: ${BACKUP_DIR}"
}

confirm_execution() {
    if [ "$DRY_RUN" = true ]; then
        return
    fi
    
    print_header "⚠️  WARNING ⚠️"
    echo "This operation will:"
    echo "  • Rewrite the entire git history"
    echo "  • Remove sensitive files permanently"
    echo "  • Remove large binary files permanently"
    echo "  • Require force-push to remote"
    echo "  • Require all collaborators to re-clone"
    echo ""
    print_warning "A backup will be created at: ${BACKUP_DIR}"
    echo ""
    read -p "Do you want to continue? (type 'yes' to proceed): " -r
    echo
    if [[ ! $REPLY =~ ^yes$ ]]; then
        print_info "Cleanup cancelled"
        exit 0
    fi
}

# =====================================
# Main Execution
# =====================================

main() {
    cd "${REPO_DIR}"
    
    print_header "FridayAI Repository Cleanup"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "Running in DRY RUN mode - no changes will be made"
    fi
    
    # Show current state
    show_file_sizes
    
    # Get confirmation
    confirm_execution
    
    # Check prerequisites
    if check_prerequisites; then
        USE_BFG=true
    else
        USE_BFG=false
    fi
    
    # Create backup before any changes
    create_backup
    
    # Download BFG if needed and available
    if [ "$USE_BFG" = true ]; then
        download_bfg
        cleanup_with_bfg
    else
        cleanup_with_filter_branch
    fi
    
    # Final cleanup
    cleanup_refs_and_gc
    
    # Show results
    show_results
    
    # Show next steps
    show_next_steps
}

# Run main function
main
