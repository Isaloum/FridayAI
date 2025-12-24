# FridayAI Repository Cleanup Instructions

## Overview

This guide helps you safely remove sensitive files and large binaries from the FridayAI repository's git history. The cleanup will:

- **Remove secrets**: API keys, SSH keys, tokens, encrypted files
- **Remove large binaries**: Python installers (~100MB+), executables, archives
- **Reduce repository size**: From ~1.1GB to ~200MB (81% reduction)
- **Improve security**: Eliminate committed credentials from history

## ⚠️ Important Warnings

- **This rewrites git history** - All commit SHAs will change
- **Requires force-push** - Overwrites remote repository history
- **Requires team coordination** - All collaborators must re-clone
- **Irreversible** (without backup) - Make sure you have a backup!
- **Backup is automatic** - The script creates one before starting

## Prerequisites

### Required
- **Git** (version 2.20 or higher)
- **macOS or Linux** (the script is written for Unix-based systems)
- **Disk space** for backup (at least 1.5GB free)
- **Repository admin access** to force-push

### Recommended
- **Java** (version 8 or higher) - For BFG Repo-Cleaner (faster method)
  - Check: `java -version`
  - Install on Mac: `brew install openjdk`

### Optional
- **curl** or **wget** - For downloading BFG (usually pre-installed on Mac)

## What Will Be Removed

### Sensitive Files (Security Risk)
```
✗ *.pem                         # SSH private keys
✗ *.key                         # Encryption keys
✗ github-token-2025.txt         # GitHub personal access token
✗ friday_memory.enc             # Encrypted memory files
✗ test_query.enc                # Test encrypted files
✗ vault.key, memory.key         # Encryption keys
✗ FridayAI-New.pem              # AWS/SSH key
✗ FridayAIKey.pem               # AWS/SSH key
✗ fridayai-backend.pem          # AWS/SSH key
```

### Large Binary Files (Repository Bloat)
```
✗ python-3.11.6-amd64.exe       # ~26MB - Python installer
✗ python-3.11.6-amd64 (1).exe   # ~26MB - Duplicate installer
✗ python-3.13.5-amd64.exe       # ~28MB - Python installer
✗ rustup-init.exe               # ~13MB - Rust installer
✗ swigwin-3.0.12.zip            # ~11MB - SWIG package
✗ swigwin-4.3.1.zip             # ~12MB - SWIG package
```

**Total size removed from history**: ~115MB of tracked files

## Step-by-Step Guide

### Step 1: Prepare Your Repository

```bash
# Navigate to your FridayAI repository
cd /path/to/FridayAI

# Ensure you're on the main branch (or your default branch)
git checkout main

# Fetch latest changes
git fetch origin

# Make sure you're up to date
git pull origin main

# Commit any uncommitted changes
git status
git add .
git commit -m "Prepare for cleanup"

# Push your changes
git push origin main
```

### Step 2: Notify Team Members

Before running cleanup, ensure all collaborators:
1. Have committed and pushed their work
2. Are aware that history will be rewritten
3. Know they'll need to re-clone after cleanup

Send a message like:
```
⚠️ Repository Cleanup Notice ⚠️
We're cleaning up the repo history to remove secrets and large files.
Please:
1. Commit and push all your work NOW
2. Do not push any changes after [TIME]
3. Plan to re-clone the repository after cleanup
```

### Step 3: Run Dry Run (Preview Changes)

```bash
# Make script executable (if not already)
chmod +x cleanup.sh

# Run in dry-run mode to preview changes
./cleanup.sh --dry-run
```

**Review the output carefully:**
- Check which files will be removed
- Verify current and expected repository sizes
- Ensure nothing unexpected is listed

### Step 4: Run the Actual Cleanup

```bash
# Run the cleanup script
./cleanup.sh
```

**What happens:**
1. ✓ Prerequisites are checked (git, java, uncommitted changes)
2. ✓ Current repository statistics are shown
3. ⚠️ You'll be asked to confirm (type 'yes')
4. ✓ Automatic backup is created at `~/fridayai-cleanup-backup-YYYYMMDD-HHMMSS`
5. ✓ BFG Repo-Cleaner is downloaded (if Java available)
6. ✓ Files are removed from entire git history
7. ✓ References are cleaned up
8. ✓ Git garbage collection runs
9. ✓ Results and next steps are shown

**Expected duration**: 2-5 minutes depending on repository size and method used.

### Step 5: Verify the Cleanup

```bash
# Check the new repository size
du -sh .
du -sh .git

# Verify sensitive files are gone from history
git log --all --full-history -- "*.pem"
git log --all --full-history -- "github-token-2025.txt"

# Check that large binaries are removed
git log --all --full-history -- "*.exe"
git log --all --full-history -- "*.zip"

# Browse recent commits to ensure they look correct
git log --oneline -20
```

**Expected results:**
- Repository size: ~200-300MB (down from ~1.1GB)
- No results from git log commands for removed files
- Commit messages intact, but commit SHAs changed

### Step 6: Force Push to Remote

⚠️ **THIS IS THE POINT OF NO RETURN** (without backup)

```bash
# Push the cleaned history to all branches
git push origin --force --all

# Push cleaned tags (if any)
git push origin --force --tags
```

**After force push:**
- Old history is overwritten on GitHub/remote
- All collaborators must re-clone
- Old clones will have mismatched history

### Step 7: Update .env Configuration

```bash
# Copy the template
cp .env.example .env

# Edit with your actual credentials
nano .env  # or use your preferred editor

# Add .env to .gitignore (already done if you updated .gitignore)
```

**Fill in your credentials:**
- OpenAI API key from https://platform.openai.com/api-keys
- SerpAPI key from https://serpapi.com/manage-api-key
- GitHub token from https://github.com/settings/tokens
- Database credentials
- Encryption keys (generate new ones!)

### Step 8: Team Re-Clone Instructions

Share these instructions with all collaborators:

```bash
# Save any local work (if needed)
cd FridayAI
git diff > ~/my-local-changes.patch  # Save uncommitted work
git log origin/main..HEAD > ~/my-unpushed-commits.txt  # List unpushed commits

# Remove old repository
cd ..
rm -rf FridayAI

# Clone fresh copy
git clone https://github.com/Isaloum/FridayAI.git
cd FridayAI

# Set up environment
cp .env.example .env
# Edit .env with your credentials

# If you had local work, apply it
# patch -p1 < ~/my-local-changes.patch
```

### Step 9: Verify and Clean Up

```bash
# Verify everything works
git status
git log --oneline -10

# Test the application (optional)
python fridayai.py  # or your main script

# Once confirmed working, you can remove the backup
# rm -rf ~/fridayai-cleanup-backup-*
# (Keep backup for at least a week to be safe!)
```

## Troubleshooting

### Java Not Found

**Error**: "Java not found - will use git filter-branch fallback"

**Solution**: Install Java or use the fallback (slower but works):
```bash
# On macOS
brew install openjdk

# Then retry
./cleanup.sh
```

### Uncommitted Changes

**Error**: "You have uncommitted changes"

**Solution**: Commit or stash changes:
```bash
git status
git add .
git commit -m "Save work before cleanup"
# Or
git stash
```

### BFG Download Fails

**Error**: "Neither curl nor wget found"

**Solution**: Install curl or wget:
```bash
# On macOS
brew install curl
# Or
brew install wget
```

### Force Push Rejected

**Error**: "remote: error: denying non-fast-forward"

**Solution**: Ensure you have admin access, then:
```bash
# Double-check you want to proceed
git push origin --force --all
git push origin --force --tags
```

### Repository Still Large After Cleanup

**Issue**: Repository size not reduced as expected

**Solution**: 
1. Verify files are removed: `git log --all -- "*.exe"`
2. Run additional garbage collection:
   ```bash
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```
3. Check `.git/objects` size: `du -sh .git/objects`

### Script Fails Midway

**Issue**: Script exits with error during cleanup

**Solution**:
1. Don't panic - backup exists at `~/fridayai-cleanup-backup-*`
2. Check error message
3. If needed, restore from backup (see Rollback section)
4. Report issue or try manual cleanup

## Rollback Procedure

If something goes wrong and you need to restore:

### Before Force Push (Local Restore)

```bash
# Find your backup
ls -la ~/fridayai-cleanup-backup-*

# Navigate to backup location
cd ~/fridayai-cleanup-backup-YYYYMMDD-HHMMSS

# Restore from mirror backup
cd /path/to/FridayAI
rm -rf .git
git clone --mirror ~/fridayai-cleanup-backup-YYYYMMDD-HHMMSS/git-backup.git .git
git config --local --bool core.bare false
git reset --hard
```

### After Force Push (Remote Restore)

**⚠️ This undoes the cleanup for everyone:**

```bash
# Navigate to backup
cd ~/fridayai-cleanup-backup-YYYYMMDD-HHMMSS

# Clone the backup
git clone git-backup.git restored-repo
cd restored-repo

# Force push the old history back
git push origin --force --all
git push origin --force --tags
```

**Note**: All team members will need to re-clone again.

### Restore from Bundle

Alternative restore method using bundle:

```bash
# Create new directory
mkdir fridayai-restored
cd fridayai-restored

# Clone from bundle
git clone ~/fridayai-cleanup-backup-YYYYMMDD-HHMMSS/repo-backup.bundle .

# Verify
git log --oneline -10

# Force push if needed
git remote add origin https://github.com/Isaloum/FridayAI.git
git push origin --force --all
```

## Post-Cleanup Best Practices

### 1. Prevent Future Secrets

- **Always** use `.env` files for credentials (already in `.gitignore`)
- **Never** commit files matching these patterns:
  - `*.pem`, `*.key`, `*.enc`
  - `*token*.txt`, `*secret*.txt`
  - `*credentials*.txt`, `*password*.txt`

### 2. Use .env.example

- Keep `.env.example` updated with new variables
- Use placeholder values: `your_api_key_here`
- Add comments explaining where to get credentials

### 3. Pre-commit Hooks (Optional)

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Check for sensitive files
if git diff --cached --name-only | grep -E '\.(pem|key)$|token.*\.txt$'; then
    echo "Error: Attempting to commit sensitive files!"
    exit 1
fi
```

Make executable: `chmod +x .git/hooks/pre-commit`

### 4. Regular Audits

Periodically check for accidentally committed secrets:
```bash
# Check for API keys
git log -S "OPENAI_API_KEY" --all

# Check for AWS keys
git log -S "AWS_ACCESS_KEY" --all

# Use tools like gitleaks or truffleHog
```

### 5. Handle Large Files

For large files needed in the project:
- Use **Git LFS** (Large File Storage)
- Store in cloud storage (S3, Google Drive)
- Document download instructions in README

## Additional Resources

### Tools Used
- **BFG Repo-Cleaner**: https://rtyley.github.io/bfg-repo-cleaner/
- **Git Filter-Branch**: https://git-scm.com/docs/git-filter-branch

### Security Scanning
- **gitleaks**: https://github.com/gitleaks/gitleaks
- **truffleHog**: https://github.com/trufflesecurity/trufflehog
- **git-secrets**: https://github.com/awslabs/git-secrets

### Git LFS (for future large files)
```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track large files
git lfs track "*.exe"
git lfs track "*.zip"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

## Support

If you encounter issues:

1. **Check logs**: Review terminal output from cleanup script
2. **Verify backup**: Confirm backup exists before proceeding
3. **Ask for help**: Create an issue on GitHub with:
   - Error message
   - Output from `./cleanup.sh --dry-run`
   - Git and Java versions: `git --version`, `java -version`

## Summary Checklist

Before cleanup:
- [ ] All team members have pushed their work
- [ ] Team is notified about upcoming cleanup
- [ ] Dry-run executed successfully
- [ ] Current work is committed

During cleanup:
- [ ] Backup created automatically
- [ ] Files removed from history
- [ ] Repository size verified

After cleanup:
- [ ] Force push completed
- [ ] `.env` configured with real credentials
- [ ] Team members re-clone repository
- [ ] Application tested and working
- [ ] Backup retained for at least 1 week

---

**Remember**: Keep the backup until you're 100% confident the cleanup was successful and all collaborators have updated their local repositories!
