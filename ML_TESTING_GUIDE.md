# ML vs A* Performance Testing Guide

## Quick Start

After the browser loads and the page is ready, open the browser console (F12 or Ctrl+Shift+K) and run:

### Quick Test (10 runs - ~1 minute)
```javascript
await window.quickMLTest()
```

### Full Test (80 runs across 4 layouts - ~4-5 minutes)
```javascript
await window.fullMLTest()
```

### Custom Test
```javascript
await window.runMLPerformanceTests(numTests, layoutType)
// Examples:
await window.runMLPerformanceTests(10, 'random')
await window.runMLPerformanceTests(20, 'maze')
await window.runMLPerformanceTests(15, 'clustered')
```

## What to Expect

The test will:
1. Generate random grid layouts
2. Create random start/goal pairs
3. Run both A* and ML heuristics
4. Compare:
   - Nodes expanded
   - Runtime (ms)
   - Path distance
   - Success rate

## Success Criteria

✓ **ML performs well if**: Nodes expanded ≤ 120% of A*'s nodes (avg)
✓ **ML performs great if**: Nodes expanded ≤ 110% of A*'s nodes or better

## Output Format

Each run shows:
```
Test 1/10: Start=[14,1] End=[5,14]
  A*:       79 nodes, 2.34ms, distance=22.40
  ML:       89 nodes, 3.12ms, distance=22.40
  ✓ ML better (ML is 112.7% of A*)
```

Summary statistics include:
- Min/Max/Average/Median nodes expanded
- Min/Max/Average/Median runtime
- Overall success rates
- Performance comparison percentage

## Troubleshooting

If the test hangs:
1. Press Ctrl+C to stop
2. Check browser console for errors
3. Verify ML model loaded: `console.log(window.staticModel)`

If ML fails but A* succeeds:
1. Check ML heuristic errors: `window.toggleMlHeuristicDebug(true)`
2. Run a quick test: `await window.quickMLTest()`

## Files Modified

- `index.html` - Added ml-performance-test.js script
- `ml-performance-test.js` - New test suite with multiple test functions
