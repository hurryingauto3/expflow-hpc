# ExpFlow Documentation Guide

This document explains the ExpFlow documentation structure.

## Documentation Files

### 1. README.md (9.4K) - Start Here
**Purpose**: Project overview and quick start guide

**Contents**:
- Quick installation and setup
- Key features overview
- Core commands reference
- "Before/After" comparison showing value
- Links to detailed documentation

**When to use**: First time users, GitHub visitors, quick reference

### 2. USER_GUIDE.md (20K) - Complete Guide
**Purpose**: Comprehensive user documentation

**Contents**:
- Detailed installation instructions
- Complete command reference with examples
- Interactive initialization walkthrough
- Experiment management workflows
- Resource and partition management
- Creating custom managers tutorial
- Advanced usage patterns
- Troubleshooting guide

**When to use**: Learning the system, implementing features, solving problems

### 3. QUICK_REFERENCE.md (4.6K) - Cheat Sheet
**Purpose**: Quick command lookup

**Contents**:
- All commands with syntax
- Common workflows
- GPU/partition types
- Status values
- Troubleshooting quick tips
- Useful bash aliases

**When to use**: Daily usage, command lookup, quick reminders

### 4. CHANGELOG.md (8.3K) - Version History
**Purpose**: Track changes across versions

**Contents**:
- Version numbers and dates
- New features per version
- Bug fixes and improvements
- Breaking changes

**When to use**: Understanding what changed, migration planning

### 5. CONTRIBUTING.md (2.4K) - For Contributors
**Purpose**: Guide for code contributors

**Contents**:
- How to contribute
- Code style guidelines
- Pull request process
- Development setup

**When to use**: Contributing code or documentation

## Removed Files (Consolidated)

The following files were **removed** to reduce documentation clutter:

- ~~INTERACTIVE_INIT_GUIDE.md~~ → Merged into USER_GUIDE.md
- ~~PARTITION_VALIDATOR_USAGE.md~~ → Merged into USER_GUIDE.md
- ~~PROJECT_STRUCTURE.md~~ → Unnecessary
- ~~TESTING.md~~ → Merged into CONTRIBUTING.md
- ~~CLAUDE.md~~ → Development-specific, not user-facing

## Documentation Flow

### New Users
1. Start with **README.md** for quick overview
2. Run `expflow init -i my-research` for interactive setup
3. Refer to **QUICK_REFERENCE.md** for daily commands
4. Read **USER_GUIDE.md** sections as needed

### Experienced Users
- **QUICK_REFERENCE.md** for command lookup
- **USER_GUIDE.md** for advanced features
- **CHANGELOG.md** for updates

### Contributors
- **CONTRIBUTING.md** for guidelines
- **CHANGELOG.md** to understand history

## Finding Information

### "How do I install ExpFlow?"
→ README.md → Installation section

### "How do I use interactive init?"
→ USER_GUIDE.md → Interactive Initialization section

### "What's the syntax for viewing logs?"
→ QUICK_REFERENCE.md → Logs section

### "What changed in v0.3.4?"
→ CHANGELOG.md → [0.3.4] section

### "How do I contribute?"
→ CONTRIBUTING.md

### "How do I create a custom manager?"
→ USER_GUIDE.md → Creating Custom Managers section

### "What are the available commands?"
→ QUICK_REFERENCE.md → Core Commands section

## Best Practices

1. **Keep README.md concise** - It's the first impression
2. **Make USER_GUIDE.md comprehensive** - But well-organized with ToC
3. **Update QUICK_REFERENCE.md** - When adding new commands
4. **Document in CHANGELOG.md** - Every version change
5. **Link between docs** - Cross-reference related content

## For Maintainers

When adding features:
1. Update USER_GUIDE.md with detailed explanation
2. Add command to QUICK_REFERENCE.md
3. Mention in README.md if it's a major feature
4. Document in CHANGELOG.md
5. Update version numbers in setup.py and __init__.py

## File Sizes Summary

```
README.md          9.4K   (Quick overview)
USER_GUIDE.md       20K   (Complete guide)
QUICK_REFERENCE.md 4.6K   (Command cheat sheet)
CHANGELOG.md       8.3K   (Version history)
CONTRIBUTING.md    2.4K   (For contributors)
──────────────────────────
Total:            44.7K   (Was 70K+ before consolidation)
```

## Documentation Principles

1. **No Duplication**: Each piece of information lives in ONE place
2. **Clear Purpose**: Each document has a specific role
3. **Easy Navigation**: Table of contents, clear sections
4. **Examples**: Show, don't just tell
5. **Up-to-date**: Update docs with code changes

---

**For complete documentation, see USER_GUIDE.md**
