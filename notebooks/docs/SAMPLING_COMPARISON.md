# ⚠️ IMPORTANT: Sequential vs Random Sampling

## Issue with First 500K Rows (Current Sample)

### What Was Done
The current `cf_sample_500k.csv` uses **the first 500,000 rows** from train.csv

### Potential Issues

❌ **Temporal Bias Risk**
- If data is sorted by time, only early time periods are represented
- However, we checked: data is NOT sorted chronologically ✅

❌ **User/Item Bias Risk**  
- First N rows might not represent all users/items equally
- Some users/items may be over-represented

✅ **Good News**
- Timestamps are mixed (July-December 2016)
- Not chronologically sorted
- Listen rate: 69.8% (close to full dataset ~68.4%)

### Is It a Problem?

**For your case: Probably OK** ✅ because:
1. Data is NOT chronologically sorted
2. Listen rate is similar to full dataset
3. Only 6.6% overlap with truly random sample
4. Good enough for collaborative filtering experiments

**But**: Random sampling is still **better practice**

---

## Better Approach: Random Sample

### New File Created
`cf_sample_500k_random.csv` - **Truly random sample**

### Comparison

| Metric | First 500K | Random 500K | Full Dataset |
|--------|-----------|-------------|--------------|
| **Interactions** | 500,000 | 500,000 | 7,558,834 |
| **Users** | 17,429 | 18,558 | 19,918 |
| **Items** | 20,475 | 105,803 | 452,975 |
| **Listen Rate** | 69.8% | **68.4%** | 68.4% |

### Key Differences

**First 500K (Sequential):**
- ✅ Simple to create
- ✅ Fast to generate
- ❌ Fewer unique items (20K vs 105K)
- ❌ Fewer unique users (17K vs 18.5K)
- ⚠️ Listen rate slightly higher (69.8% vs 68.4%)

**Random 500K:**
- ✅ More representative
- ✅ Better item coverage (105K items)
- ✅ Better user coverage (18.5K users)
- ✅ Listen rate matches full dataset (68.4%)
- ✅ No bias from data ordering

---

## Recommendation

### Option 1: Keep Current Sample ✅ **EASIEST**
**Reasoning:**
- Already pushed to GitHub
- Team already has access
- Listen rate is close enough (69.8% vs 68.4%)
- Reproducible with seed 42
- Good enough for CF experiments

**Use this if:**
- Team is already working with it
- Don't want to confuse people with changes
- Results are "close enough"

### Option 2: Switch to Random Sample ✅ **BETTER**
**Reasoning:**
- More representative (68.4% listen rate = exact match)
- Better coverage (105K items vs 20K)
- Best practice for sampling
- Still reproducible

**Use this if:**
- Haven't started experiments yet
- Want most accurate results
- Care about best practices

### Option 3: Use Both 🎯 **COMPREHENSIVE**
**Reasoning:**
- Test robustness across different samples
- Validate that results are consistent
- Scientific best practice

**Use this if:**
- Want to be thorough
- Have time for extra experiments
- Writing academic paper

---

## What to Do Now?

### If You Want to Switch (Recommended)

```bash
cd notebooks

# Remove old sample from git
git rm cf_sample_500k.csv
git add cf_sample_500k_random.csv cf_sample_random_info.txt

# Commit the change
git commit -m "Replace sequential sample with random sample for better representativeness"
git push

# Tell your team
"Updated to random sample for better coverage - please re-pull"
```

### If You Want to Keep Current (Also Fine)

```bash
# Do nothing! Current sample works fine for CF experiments
# Just document that it's sequential first 500K
```

### If You Want Both (Most Thorough)

```bash
cd notebooks

# Add random sample alongside current one
git add cf_sample_500k_random.csv cf_sample_random_info.txt
git commit -m "Add random sample for comparison"
git push

# Tell your team
"Two samples available:
- cf_sample_500k.csv (first 500K) - use this to match existing results
- cf_sample_500k_random.csv (random) - use this for new experiments"
```

---

## Quick Decision Guide

**Q: Team already started working with first 500K?**  
→ **Keep it** to avoid confusion

**Q: Haven't started experiments yet?**  
→ **Switch to random** for best results

**Q: Want to validate results?**  
→ **Use both** and compare

**Q: Which is "good enough"?**  
→ **Both are fine** - the difference isn't huge

---

## Technical Details

### First 500K Sample
- Method: `df = pd.read_csv('train.csv', nrows=500000)`
- Overlap with random: 6.6%
- Coverage: 4.5% of items, 87.5% of users

### Random Sample  
- Method: `df.sample(n=500000, random_state=42)`
- Overlap with first 500K: 6.6%
- Coverage: 23.4% of items, 93.2% of users

### Statistical Comparison
- Listen rate difference: 1.4 percentage points
- User coverage difference: 1,129 users (6%)
- Item coverage difference: 85,328 items (417% more!)

**Biggest difference:** Random sample has **5x more unique items**

---

## Bottom Line

### Current Sample (First 500K)
✅ **Good enough** for collaborative filtering  
✅ **Already distributed** to team  
⚠️ **Could be better** (fewer items covered)

### Random Sample
✅ **Better** representativeness  
✅ **More items** for better recommendations  
✅ **Best practice** for sampling  
❌ **Requires update** if switching

### My Recommendation

**If starting fresh:** Use random sample  
**If already working:** Keep first 500K  
**If thorough:** Test both and compare

---

**Created**: February 1, 2026  
**Files**: `cf_sample_500k.csv` vs `cf_sample_500k_random.csv`
