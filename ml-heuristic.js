// ML-based heuristics using TensorFlow.js models exported to web_model_{static,dynamic}
// These functions attempt to load a TFJS model and predict a residual to add to an
// octile distance heuristic. If the model can't be loaded, they fall back to octile.

let staticModelPromise = null;
let dynamicModelPromise = null;

// Global counters for diagnosing issues
let ML_CALLS = 0;
let ML_MODEL_LOADS = 0;
let ML_PREDICTION_FAILURES = 0;
let ML_VALID_RESIDUALS = 0;

// Expose diagnostic info to window
if (typeof window !== 'undefined') {
    window.getMLDiagnostics = () => ({
        totalCalls: ML_CALLS,
        modelLoads: ML_MODEL_LOADS,
        predictionFailures: ML_PREDICTION_FAILURES,
        validResiduals: ML_VALID_RESIDUALS,
        failureRate: ML_CALLS > 0 ? (ML_PREDICTION_FAILURES / ML_CALLS * 100).toFixed(1) + '%' : 'N/A'
    });
    window.resetMLDiagnostics = () => {
        ML_CALLS = ML_MODEL_LOADS = ML_PREDICTION_FAILURES = ML_VALID_RESIDUALS = 0;
        console.log('ML diagnostics reset');
    };
}

// Log module load for easier debugging in browser console
try {
    if (typeof window !== 'undefined' && window.console) console.log('[ml-heuristic] module loaded');
} catch (e) {}

// Debug flag: set to true to log model load and prediction details
let DEBUG_ML_HEURISTIC = false;  // Set to true to debug regressions
let DEBUG_PRED_COUNT = 0;
const MAX_DEBUG_PREDS = 100; // Log first N predictions
// Expose toggle to window for quick testing
if (typeof window !== 'undefined') {
    window.toggleMlHeuristicDebug = (v) => { DEBUG_ML_HEURISTIC = !!v; DEBUG_PRED_COUNT = 0; console.log('ML heuristic debug:', DEBUG_ML_HEURISTIC); };
}

function octileDistance(a, b) {
    const dx = Math.abs(a.row - b.row);
    const dy = Math.abs(a.col - b.col);
    const D = 1;
    const D2 = Math.SQRT2;
    return D * (dx + dy) + (D2 - 2 * D) * Math.min(dx, dy);
}

async function loadModel(path) {
    if (!window || !window.tf) return null;
    try {
        // tf.loadGraphModel works for models converted with the tfjs_graph_model converter
        const model = await window.tf.loadGraphModel(path);
        return model;
    } catch (e) {
        // Try layers model as a fallback
        try {
            const model = await window.tf.loadLayersModel(path);
            return model;
        } catch (err) {
            console.warn('Failed to load TFJS model at', path, err);
            return null;
        }
    }
}

async function getStaticModel() {
    if (!staticModelPromise) {
        if (DEBUG_ML_HEURISTIC) console.log('[getStaticModel] Loading model for first time...');
        staticModelPromise = loadModel('./web_model_static/model.json').then(m => {
            ML_MODEL_LOADS++;
            if (DEBUG_ML_HEURISTIC) console.log('[getStaticModel] Model load completed:', !!m);
            return m;
        }).catch(e => {
            if (DEBUG_ML_HEURISTIC) console.error('[getStaticModel] Model load failed:', e.message);
            staticModelPromise = null; // Reset so we can retry
            throw e;
        });
    }
    const m = await staticModelPromise;
    if (DEBUG_ML_HEURISTIC) console.log('[getStaticModel] Returning cached model:', !!m);
    return m;
}

async function getDynamicModel() {
    if (!dynamicModelPromise) dynamicModelPromise = loadModel('./web_model_dynamic/model.json');
    const m = await dynamicModelPromise;
    if (DEBUG_ML_HEURISTIC) console.log('Dynamic model loaded:', !!m);
    return m;
}

function gridToArray(grid) {
    // Expect grid to be an array of rows of cell objects with .isWall truthy for obstacles
    if (!grid || !Array.isArray(grid) || grid.length === 0) {
        console.error('[ML] gridToArray: invalid grid input', typeof grid, Array.isArray(grid), grid?.length);
        return null;
    }
    const rows = grid.length;
    const cols = grid[0].length;
    
    // Validate grid structure
    if (!Array.isArray(grid[0])) {
        console.error('[ML] gridToArray: grid[0] is not an array, is:', typeof grid[0]);
        return null;
    }
    
    const arr = new Float32Array(rows * cols);
    let wallCount = 0;
    let errors = [];
    
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = grid[r][c];
            if (!cell) {
                if (errors.length < 5) errors.push(`grid[${r}][${c}] is null`);
                arr[r * cols + c] = 0.0;
                continue;
            }
            if (typeof cell.isWall === 'undefined') {
                if (errors.length < 5) errors.push(`grid[${r}][${c}].isWall is undefined`);
                arr[r * cols + c] = 0.0;
                continue;
            }
            arr[r * cols + c] = cell.isWall ? 1.0 : 0.0;
            if (cell.isWall) wallCount++;
        }
    }
    
    if (errors.length > 0 && DEBUG_ML_HEURISTIC) {
        console.log('[ML] gridToArray errors:', errors);
    }
    
    if (DEBUG_ML_HEURISTIC) {
        // Log grid structure info
        const wallIndices = [];
        for (let i = 0; i < Math.min(arr.length, 400); i++) {
            if (arr[i] > 0.5) wallIndices.push(i);
        }
        // Show a few sample cells for validation
        const sample00 = grid[0][0];
        const sample19 = grid[0][19];
        const sample1919 = grid[19][19];
        console.log(`[ML] gridToArray: rows=${rows}, cols=${cols}, totalCells=${rows*cols}, wallCount=${wallCount}`);
        console.log(`[ML] grid[0][0]={isWall:${sample00?.isWall}, row:${sample00?.row}, col:${sample00?.col}}`);
        console.log(`[ML] grid[0][19]={isWall:${sample19?.isWall}, row:${sample19?.row}, col:${sample19?.col}}`);
        console.log(`[ML] grid[19][19]={isWall:${sample1919?.isWall}, row:${sample1919?.row}, col:${sample1919?.col}}`);
        console.log(`[ML] wallIndices (first 20): [${wallIndices.slice(0, 20).join(',')}]`);
    }
    
    return { arr, rows, cols };
}

async function predictResidual(model, gridObj, start, goal) {
    if (!model || !window.tf) return 0;
    const tf = window.tf;
    const { arr, rows, cols } = gridObj;
    
    if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
        // Log first few predictions with grid stats
        const wallCount = Array.from(arr).filter(v => v > 0).length;
        console.log(`[ML pred] Grid stats: ${wallCount} walls, first 5 values: [${Array.from(arr.slice(0, 5)).map(v => v.toFixed(1)).join(',')}]`);
    }
    
    try {
        const gridTensor = tf.tensor4d(arr, [1, rows, cols, 1], 'float32');
        const dist = octileDistance(start, goal);
        const maxDist = Math.sqrt(2) * (rows - 1); // Assume square grid (rows === cols)
        // Use only 4 features: start_r, start_c, goal_r, goal_c (normalized)
        // Drop the normalized distance feature if the model expects only 4
        const sg = [start.row / (rows - 1), start.col / (cols - 1), goal.row / (rows - 1), goal.col / (cols - 1)];
        const sgTensor = tf.tensor2d([sg], [1, 4], 'float32');
        
        if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
            console.log(`[ML pred] gridTensor shape: ${gridTensor.shape}, sgTensor shape: ${sgTensor.shape}`);
            console.log(`[ML pred] start/goal normalized (4 features): [${sg.map(v => v.toFixed(3)).join(',')}]`);
        }

        let out = null;
        let method = 'unknown';
        
        // Log model input names if available
        if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
            if (model.inputNames) console.log(`[ML pred] Model inputNames: [${model.inputNames.join(',')}]`);
            if (model.outputs) console.log(`[ML pred] Model outputNames: [${model.outputNames ? model.outputNames.join(',') : 'unknown'}]`);
        }
        
        // Try execute() with the known input names from the model signature
        // The TFJS model has inputs: {grid, start_goal}
        try {
            if (model.execute) {
                out = model.execute({grid: gridTensor, start_goal: sgTensor});
                if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                    console.log(`[ML pred] model.execute({grid, start_goal}) succeeded`);
                }
            } else if (model.predict) {
                // Fallback for layers models
                out = model.predict([gridTensor, sgTensor]);
                if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                    console.log(`[ML pred] model.predict() succeeded`);
                }
            }
        } catch (execErr) {
            if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                console.log(`[ML pred] execute/predict failed: ${execErr.message.split('\n')[0]}`);
            }
        }

        if (!out) {
            gridTensor.dispose();
            sgTensor.dispose();
            if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                console.log(`[ML pred] no output from model, tried execute with multiple input names`);
            }
            return 0;
        }

        const predVal = (await (Array.isArray(out) ? out[0].data() : out.data()))[0];
        const max_cost = Math.SQRT2 * (rows - 1);
        const predicted_cost = predVal * max_cost;  // Model predicts normalized cost [0,1]
        const octile = octileDistance(start, goal);
        const residual = predicted_cost - octile;   // Residual = cost - heuristic
        
        // Validate prediction
        if (!isFinite(predVal) || !isFinite(residual)) {
            ML_PREDICTION_FAILURES++;
            if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                DEBUG_PRED_COUNT++;
                console.warn(`[ML pred #${DEBUG_PRED_COUNT}] INVALID PREDICTION! predVal=${predVal}, residual=${residual}, returning 0`);
            }
            try { gridTensor.dispose(); } catch (e) {}
            try { sgTensor.dispose(); } catch (e) {}
            try { if (out.dispose) out.dispose(); } catch (e) {}
            return 0;
        }
        
        ML_VALID_RESIDUALS++;
        if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
            DEBUG_PRED_COUNT++;
            const actualWallCount = Array.from(arr).filter(v => v > 0.5).length;
            console.log(`[ML pred #${DEBUG_PRED_COUNT}] walls=${actualWallCount}/400, start=[${start.row},${start.col}], goal=[${goal.row},${goal.col}], predVal=${predVal.toFixed(4)}, predicted_cost=${predicted_cost.toFixed(3)}, octile=${octile.toFixed(3)}, residual=${residual.toFixed(3)}`);
        }

        try { gridTensor.dispose(); } catch (e) {}
        try { sgTensor.dispose(); } catch (e) {}
        try { if (out.dispose) out.dispose(); } catch (e) {}

        return residual;
    } catch (err) {
        if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
            console.log(`[ML pred] error: ${err.message}`);
        }
        return 0;
    }
}

async function mlHeuristic(current, goal, grid) {
    // If no grid provided, fallback to octile distance
    ML_CALLS++;
    if (!grid || !Array.isArray(grid)) return octileDistance(current, goal);
    if (DEBUG_ML_HEURISTIC) console.log('[mlHeuristic] called (call #' + ML_CALLS + '), loading model...');
    const model = await getStaticModel();
    if (!model) {
        ML_PREDICTION_FAILURES++;
        if (DEBUG_ML_HEURISTIC) console.error('[mlHeuristic] MODEL IS NULL - CRITICAL ERROR! Falling back to octile');
        return octileDistance(current, goal);
    }
    const gridObj = gridToArray(grid);
    const residual = await predictResidual(model, gridObj, current, goal);
    const result = octileDistance(current, goal) + residual;
    if (DEBUG_ML_HEURISTIC && Math.random() < 0.05) console.log('[mlHeuristic] result =', result, 'residual =', residual); // Log 5% of calls
    return result;
}

async function mlDynamicHeuristic(current, goal, grid) {
    if (!grid || !Array.isArray(grid)) return octileDistance(current, goal);
    if (DEBUG_ML_HEURISTIC) console.log('[mlDynamicHeuristic] called');
    const model = await getDynamicModel();
    if (!model) {
        if (DEBUG_ML_HEURISTIC) console.log('[mlDynamicHeuristic] model is null, using octile fallback');
        return octileDistance(current, goal);
    }
    const gridObj = gridToArray(grid);
    const residual = await predictResidual(model, gridObj, current, goal);
    const result = octileDistance(current, goal) + residual;
    if (DEBUG_ML_HEURISTIC) console.log('[mlDynamicHeuristic] result =', result);
    return result;
}

export { mlHeuristic, mlDynamicHeuristic };
