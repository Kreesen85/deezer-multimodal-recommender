# Adding CF Sample to GitHub - Quick Guide

## ✅ Good News!

The `cf_sample_500k.csv` file is **34 MB**, which is:
- ✅ **Under GitHub's 100 MB limit** (you're good to push!)
- ✅ **Under GitHub's 50 MB warning threshold** (no warnings)

You can safely add it to your repository!

---

## 🚀 How to Add to GitHub

### Step 1: Stage the Files

```bash
cd /Users/kreesen/Documents/deezer-multimodal-recommender

# Add the sample file and documentation
git add notebooks/cf_sample_500k.csv
git add notebooks/cf_sample_info.txt
git add notebooks/CF_SAMPLE_TEAM_GUIDE.md
git add notebooks/README_CF_SAMPLE.md
git add notebooks/create_cf_sample.py
git add TEAM_COLLABORATION_SETUP.md

# Check what will be committed
git status
```

### Step 2: Commit

```bash
git commit -m "Add reproducible CF sample dataset for team collaboration

- cf_sample_500k.csv: 500K interactions sample (34 MB)
- Complete documentation and usage guides
- Random seed 42 for reproducibility
- Enables consistent results across team members"
```

### Step 3: Push to GitHub

```bash
git push origin main
# or: git push origin master
```

---

## 📥 Your Colleagues Can Then:

```bash
# Clone or pull the repo
git clone <your-repo-url>
# or
git pull

# The sample file will be in notebooks/
cd notebooks
ls -lh cf_sample_500k.csv  # Should see 34 MB file

# Use it immediately
python baseline_collaborative_filtering.py
```

---

## 🗑️ To Remove Later (if needed)

If you want to remove the file from Git later:

```bash
# Remove from Git but keep locally
git rm --cached notebooks/cf_sample_500k.csv
git commit -m "Remove CF sample from repo (use cloud storage instead)"
git push

# Then add to .gitignore to prevent re-adding
echo "notebooks/cf_sample_500k.csv" >> .gitignore
git add .gitignore
git commit -m "Ignore CF sample file"
git push
```

---

## 💡 Alternative: Git LFS (for larger files in future)

If you later have files > 100 MB, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "notebooks/*.csv"
git add .gitattributes

# Then commit normally
git add notebooks/cf_sample_500k.csv
git commit -m "Add CF sample with Git LFS"
git push
```

---

## ✅ Recommended Approach

**For now:** Just add it normally (34 MB is fine!)

```bash
git add notebooks/cf_sample_500k.csv \
        notebooks/cf_sample_info.txt \
        notebooks/CF_SAMPLE_TEAM_GUIDE.md \
        notebooks/README_CF_SAMPLE.md \
        notebooks/create_cf_sample.py \
        TEAM_COLLABORATION_SETUP.md

git commit -m "Add reproducible CF sample for team"
git push
```

**Benefits:**
- ✅ Simple - no special tools needed
- ✅ Fast - within GitHub limits
- ✅ Immediate - colleagues can pull right away
- ✅ Version controlled - tracks changes

---

## 📊 File Size Reference

| Threshold | Size | Your File | Status |
|-----------|------|-----------|--------|
| GitHub Limit | 100 MB | 34 MB | ✅ Safe |
| GitHub Warning | 50 MB | 34 MB | ✅ No warning |
| Recommended Max | 25 MB | 34 MB | ⚠️ Slightly over recommendation |

**Verdict:** You're good to go! 🚀

---

## 🎯 Summary

1. ✅ 34 MB is **safe for GitHub** (under 100 MB limit)
2. ✅ No warnings (under 50 MB threshold)
3. ✅ Use regular `git add` and `git push`
4. ✅ Your team can clone/pull immediately
5. ✅ Can remove later if you switch to cloud storage

**Go ahead and push it!** 🎉
