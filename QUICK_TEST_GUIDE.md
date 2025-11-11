# Quick Reference: Running ML Performance Tests

## ⚡ Super Quick Start

1. **Open Browser**: http://localhost:8000
2. **Open Console**: Press F12 (or Ctrl+Shift+K on Linux)
3. **Run Test**: Paste this in console and press Enter:
   ```javascript
   await window.quickMLTest()
   ```
4. **Wait**: ~1 minute for 10 test runs
5. **Check Results**: Look for the SUMMARY section at the bottom

---

## 📊 Test Options

### Option 1: Quick Test (RECOMMENDED FIRST)
```javascript
await window.quickMLTest()
```
- **Time**: ~1 minute
- **Runs**: 10 (on random layout)
- **Good for**: Initial verification

### Option 2: Full Test (COMPREHENSIVE)
```javascript
await window.fullMLTest()
```
- **Time**: ~5 minutes
- **Runs**: 80 (20 runs × 4 layouts)
- **Good for**: Complete validation

### Option 3: Custom Test
```javascript
await window.runMLPerformanceTests(numTests, layoutType)
```
**Examples**:
```javascript
await window.runMLPerformanceTests(20, 'random')    // 20 runs on random
await window.runMLPerformanceTests(15, 'maze')      // 15 runs on maze
await window.runMLPerformanceTests(25, 'clustered') // 25 runs on clustered
```

---

## ✅ What to Look For

### Success Indicators

**EXCELLENT** 🟢
```
ML avg nodes / A* avg nodes: 95-105%
✓ ML performs equal or better than A*
```

**GOOD** 🟡
```
ML avg nodes / A* avg nodes: 105-115%
✓ ML performs close to A*
```

**ACCEPTABLE** 🟠
```
ML avg nodes / A* avg nodes: 115-120%
⚠ ML is slightly slower but still reasonable
```

**NEEDS WORK** 🔴
```
ML avg nodes / A* avg nodes: >120%
✗ ML performs significantly worse than A*
```

---

## 📈 Sample Output (What You'll See)

```
========== ML vs A* Performance Test (10 runs) ==========

Test 1/10: Start=[14,1] End=[5,14]
  A*:       79 nodes, 2.34ms, distance=22.40
  ML:       89 nodes, 3.12ms, distance=22.40
  ✓ ML better (ML is 112.7% of A*)

Test 2/10: Start=[8,5] End=[18,15]
  A*:       45 nodes, 1.23ms, distance=14.66
  ML:       48 nodes, 1.89ms, distance=14.66
  ✓ ML better (ML is 106.7% of A*)

[... 8 more tests ...]

========== SUMMARY ==========

Tests completed: 10
A* successes: 10/10
ML successes: 10/10

A* Nodes Expanded:    min=45, max=156, avg=92.3, median=89
A* Runtime (ms):      min=1.23, max=5.67, avg=3.21, median=3.05

ML Nodes Expanded:    min=48, max=162, avg=101.2, median=98
ML Runtime (ms):      min=2.15, max=8.34, avg=4.56, median=4.31

========== COMPARISON ==========
ML avg nodes / A* avg nodes: 109.5%
✓ ML performs equal or better than A*
ML avg time / A* avg time: 142.1%
```

---

## 🐛 Troubleshooting

### Issue: Test hangs or never completes
**Solution**:
1. Press Ctrl+C to stop
2. Check if model loaded: `console.log(window.staticModel)`
3. Try a quick test with just 1 run: `await window.runMLPerformanceTests(1, 'random')`

### Issue: "window.runMLPerformanceTests is not defined"
**Solution**:
1. Refresh page (Ctrl+R or Cmd+R)
2. Wait 3-5 seconds for all scripts to load
3. Try again

### Issue: ML tests fail but A* works
**Solution**:
1. Enable debug logging: `window.toggleMlHeuristicDebug(true)`
2. Run one test: `await window.quickMLTest()`
3. Check console for "[ML]" debug messages

### Issue: Different results each time
**This is normal!** Each test generates random grids and start/goal pairs.
- Run full test for more stable averages
- Average of 20+ runs is more representative

---

## 💡 Understanding the Metrics

### Nodes Expanded
- **Lower is better** - fewer nodes explored = more efficient search
- ML/A* ratio should be ≤ 120% for good heuristic

### Runtime (ms)
- Affected by nodes expanded + ML model inference time
- ML may be slower even if expanding fewer nodes (model overhead)
- Acceptable if runtime ratio ≤ 150%

### Distance
- Should be **identical** for both algorithms
- Both find optimal path (A* always does, ML should too)
- Different distance = error (check logs)

### Success Rate
- Should be **100%** for both
- Failures indicate bugs (check logs with debug enabled)

---

## 🎯 Next Steps After Testing

### If Results are GOOD (ML ≤ 110% of A*):
1. ✅ ML heuristic is working well
2. Consider which layout types are best
3. Tune ML model if needed (retrain with more data)

### If Results are ACCEPTABLE (110-120%):
1. ✅ ML is functional but could be better
2. May benefit from model retraining
3. Consider ensemble with A* (use ML when confident)

### If Results are POOR (>120%):
1. Check debug logs: `window.toggleMlHeuristicDebug(true)`
2. Verify model loaded correctly
3. May need to retrain the ML model
4. Check for numerical issues in preprocessing

---

## 📝 Tips for Best Results

1. **Run on consistent hardware**: Same computer for comparison
2. **Close other programs**: Reduce system load for stable timing
3. **Run at least 20 tests**: More tests = more stable statistics
4. **Try different layouts**: ML may be better/worse on certain types
5. **Compare consistent versions**: Same code, same model weights

---

**Last Updated**: November 10, 2025
**Test Suite**: ml-performance-test.js
**Guide**: ML_TESTING_GUIDE.md
