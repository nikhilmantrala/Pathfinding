# ML Heuristic Regression Analysis & Fixes

## Problem Summary
The ML heuristic performed well on the first pathfinding test:
- **Test 1**: ML 207 nodes vs A* 218 nodes ✅ (ML wins by 11)

But then drastically degraded:
- **Test 2**: ML 108 nodes vs A* 55 nodes ❌ (ML loses by 53, nearly 2x worse!)
- **Test 3**: ML 101 nodes vs A* 29 nodes ❌ (ML loses by 72, 3.5x worse!)

Plus **severe performance issue**: ML took 3102ms vs A*'s 2.6ms (1200x slower)

## Root Cause Analysis

### What We Verified ✓
1. **SavedModel (Python)**: Predictions are valid and good
   - Tested on 50 random grids with 250 total start/goal pairs
   - 0 "terrible" grids (predictions worse than octile)
   - No NaN or invalid outputs
   - Model output range is [0, 1] as expected

2. **TFJS Conversion**: Files exist and are complete
   - model.json: 15,422 bytes
   - group1-shard1of1.bin: 1,725,808 bytes  
   - All 20 weights present in manifest

3. **Model Architecture**: Correct input signatures
   - grid: [-1, 20, 20, 1]
   - start_goal: [-1, 5]
   - output: [-1, 1]

### Likely Issues (to be tested)
1. **TFJS Model Loading**: May be silently failing and falling back to octile-only heuristic
2. **Tensor Disposal**: May be corrupting tensor state after first use
3. **Async Execution**: May have race conditions or await issues
4. **Input Name Mismatch**: The execute() call may need different input names

## Fixes Implemented

### 1. Simplified Model Execution (ml-heuristic.js)
- **Old**: Tried multiple input name combinations in a loop
- **New**: Uses the known input names {grid, start_goal} from TFJS signature
- **Why**: Reduces risk of tensor corruption from multiple execution attempts

### 2. Better Error Handling
- Added NaN/Infinity validation for predictions
- Added try/catch for model.execute() with clear fallback
- Log which execute method succeeded (execute vs predict)

### 3. Global Diagnostics
Added counters and functions to window object:
- `window.getMLDiagnostics()` - Shows current status
- `window.resetMLDiagnostics()` - Resets counters
- `window.toggleMlHeuristicDebug(bool)` - Enables detailed logging
- Tracks: totalCalls, modelLoads, predictionFailures, validResiduals, failureRate

### 4. Diagnostic HTML Page (ml-diagnostic.html)
Interactive test page with buttons to:
- Test model loading
- Test prediction accuracy
- Run full diagnostics
- Enable debug mode

## Next Steps to Verify

### Step 1: Test Model Loading
Open in browser: `ml-diagnostic.html`
Click "Test Model Load" button
Expected: "Model loaded successfully!" message

### Step 2: Test Predictions
Click "Test Prediction" button
Expected: Prediction value between 0 and 1, marked as VALID

### Step 3: Run Pathfinding Benchmarks
Open `index.html` and run the same tests:
- Test 1: start=(13,1), goal=(5,15)
- Test 2: start=(14,4), goal=(7,15)  
- Test 3: start=(17,3), goal=(8,15)

Check console with: `window.getMLDiagnostics()`
Expected: 
- totalCalls >> 0 (model is being used)
- predictionFailures == 0 (no failures)
- failureRate == "0.0%"

### Step 4: Monitor Performance
- Check if ML time approaches A* time (should be 1-3x slower, not 1200x)
- Check if ML visits fewer or equal nodes than A*

## If Issues Persist

### If Model Won't Load
- Check browser console for CORS errors
- Verify web_model_static/model.json exists and is accessible
- Check that TensorFlow.js is loaded before ml-heuristic.js

### If Predictions Are Invalid
- Check if output is NaN: `!isFinite(predVal)`
- Enable debug mode and look for error messages
- May need to rebuild TFJS model from SavedModel

### If Performance Is Still Slow
- Profile in browser DevTools
- Check if await is being properly waited on
- May need to cache predictions or batch them

### If Regression Persists
- The issue might be with the training data itself
- May need to retrain with different hyperparameters
- Consider falling back to octile-only heuristic as safe fallback
