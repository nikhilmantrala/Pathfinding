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
let DEBUG_ML_HEURISTIC = true;  // Set to true to debug regressions
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
    if (!grid || !Array.isArray(grid) || grid.length === 0) {
        console.warn('[gridToArray] Invalid grid provided');
        return { arr: new Float32Array(400), rows: 20, cols: 20, originalGrid: null }; // default empty grid
    }
    
    const rows = grid.length;
    const cols = grid[0].length;
    const arr = new Float32Array(rows * cols);
    
    let idx = 0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = grid[r][c];
            if (!cell) {
                arr[idx++] = 0;
                continue;
            }
            // Cell is a wall if isWall is true
            arr[idx++] = cell.isWall ? 1 : 0;
        }
    }
    
    return { arr, rows, cols, originalGrid: grid };
}

async function predictResidual(model, gridObj, start, goal) {
    if (!model || !globalThis.tf) return 0;
    const tf = globalThis.tf;
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
        
        // Detect model type based on input names
        const isStaticModel = model.inputNames?.includes('start_goal') && !model.inputNames?.includes('cost');
        const isDynamicModel = model.inputNames?.includes('cost');
        
        let sgTensor, costTensor = null;
        
        // All models now expect 5 features: start_r, start_c, goal_r, goal_c, normalized_distance
        const normalizedDist = dist / maxDist;
        const sg = [
            start.row / (rows - 1), 
            start.col / (cols - 1), 
            goal.row / (rows - 1), 
            goal.col / (cols - 1),
            normalizedDist
        ];
        sgTensor = tf.tensor2d([sg], [1, 5], 'float32');
        
        if (isDynamicModel) {
            // Create cost grid - extract actual costs from grid cells
            const costArr = new Float32Array(rows * cols);
            let idx = 0;
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    // Extract cost from grid structure - check both possible sources
                    const cellCost = gridObj.originalGrid?.[r]?.[c]?.cost || 1;
                    costArr[idx++] = cellCost;
                }
            }
            costTensor = tf.tensor4d(costArr, [1, rows, cols, 1], 'float32');
        }
        
        if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
            console.log(`[ML pred] gridTensor shape: ${gridTensor.shape}, sgTensor shape: ${sgTensor.shape}`);
            if (costTensor) console.log(`[ML pred] costTensor shape: ${costTensor.shape}`);
            let debugModelType = 'unknown';
            if (isStaticModel) {
                debugModelType = 'static';
            } else if (isDynamicModel) {
                debugModelType = 'dynamic';
            }
            console.log(`[ML pred] Model type: ${debugModelType}`);
        }

        let out = null;
        // Log model input names if available
        if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
            if (model.inputNames) console.log(`[ML pred] Model inputNames: [${model.inputNames.join(',')}]`);
            if (model.outputs) console.log(`[ML pred] Model outputNames: [${model.outputNames ? model.outputNames.join(',') : 'unknown'}]`);
        }
        
        // Execute model with correct inputs based on model type
        try {
            if (model.execute) {
                if (isDynamicModel && costTensor) {
                    // Dynamic model: grid, cost, start_goal
                    out = model.execute({
                        "grid:0": gridTensor, 
                        "cost:0": costTensor, 
                        start_goal: sgTensor
                    });
                } else {
                    // Static model: grid, start_goal
                    out = model.execute({grid: gridTensor, start_goal: sgTensor});
                }
                if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                    console.log(`[ML pred] model.execute() succeeded with ${isDynamicModel ? '3' : '2'} inputs`);
                }
            } else if (model.predict) {
                // Fallback for layers models
                const inputs = isDynamicModel && costTensor ? [gridTensor, costTensor, sgTensor] : [gridTensor, sgTensor];
                out = model.predict(inputs);
                if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                    console.log(`[ML pred] model.predict() succeeded`);
                }
            }
        } catch (error_) {
            if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                console.log(`[ML pred] execute/predict failed: ${error_.message.split('\n')[0]}`);
            }
        }

        if (!out) {
            // Clean up tensors before returning
            gridTensor.dispose();
            sgTensor.dispose();
            if (costTensor) costTensor.dispose();
            if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                console.log(`[ML pred] no output from model, tried execute with correct input format`);
            }
            return 0;
        }

        const predVal = (await (Array.isArray(out) ? out[0].data() : out.data()))[0];
        const max_cost = Math.SQRT2 * (rows - 1);
        const predicted_cost = predVal * max_cost;  // Model predicts normalized cost [0,1]
        const octile = octileDistance(start, goal);
        
        // CRITICAL FIX: Ensure admissibility by never overestimating
        // Use the minimum of predicted cost and octile distance to ensure admissibility
        const admissible_heuristic = Math.max(predicted_cost, octile);
        const residual = admissible_heuristic - octile;  // This will be <= 0, ensuring admissibility
        
        // Validate prediction
        if (!Number.isFinite(predVal) || !Number.isFinite(residual)) {
            ML_PREDICTION_FAILURES++;
            if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
                DEBUG_PRED_COUNT++;
                console.warn(`[ML pred #${DEBUG_PRED_COUNT}] INVALID PREDICTION! predVal=${predVal}, residual=${residual}, returning 0`);
            }
            gridTensor.dispose();
            sgTensor.dispose();
            if (costTensor) costTensor.dispose();
            if (out.dispose) out.dispose();
            return 0;
        }
        
        ML_VALID_RESIDUALS++;
        if (DEBUG_ML_HEURISTIC && DEBUG_PRED_COUNT < MAX_DEBUG_PREDS) {
            DEBUG_PRED_COUNT++;
            const actualWallCount = Array.from(arr).filter(v => v > 0.5).length;
            let modelType = 'unknown';
            if (isStaticModel) {
                modelType = 'static';
            } else if (isDynamicModel) {
                modelType = 'dynamic';
            }
            console.log(`[ML pred #${DEBUG_PRED_COUNT}] ${modelType} model, walls=${actualWallCount}/400, start=[${start.row},${start.col}], goal=[${goal.row},${goal.col}], predVal=${predVal.toFixed(4)}, predicted_cost=${predicted_cost.toFixed(3)}, octile=${octile.toFixed(3)}, residual=${residual.toFixed(3)}`);
        }

        // Clean up tensors
        gridTensor.dispose();
        sgTensor.dispose();
        if (costTensor) costTensor.dispose();
        if (out.dispose) out.dispose();

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

// Add diagnostic function for testing model inputs and predictions
async function testMLModelDiagnostics(testGrid, start, goal) {
    if (typeof globalThis !== 'undefined' && globalThis.console) {
        console.log('=== ML Model Diagnostics ===');
        const staticModel = await getStaticModel();
        const dynamicModel = await getDynamicModel();
        
        const gridObj = gridToArray(testGrid);
        
        console.log(`Grid dimensions: ${gridObj.rows}x${gridObj.cols}`);
        console.log(`Start: [${start.row}, ${start.col}], Goal: [${goal.row}, ${goal.col}]`);
        console.log(`Octile distance: ${octileDistance(start, goal).toFixed(3)}`);
        
        if (staticModel) {
            console.log('\n--- Static Model ---');
            console.log('Input names:', staticModel.inputNames);
            const staticResidual = await predictResidual(staticModel, gridObj, start, goal);
            console.log(`Static residual: ${staticResidual.toFixed(3)}`);
        }
        
        if (dynamicModel) {
            console.log('\n--- Dynamic Model ---');
            console.log('Input names:', dynamicModel.inputNames);
            const dynamicResidual = await predictResidual(dynamicModel, gridObj, start, goal);
            console.log(`Dynamic residual: ${dynamicResidual.toFixed(3)}`);
        }
        
        console.log('\n=== End Diagnostics ===');
    }
}

// Expose to window for easy testing
if (typeof globalThis !== 'undefined') {
    globalThis.testMLModelDiagnostics = testMLModelDiagnostics;
}

export { mlHeuristic, mlDynamicHeuristic };
