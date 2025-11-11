// Quick test of the TFJS model via Node.js
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const GRID_SIZE = 20;

function octileDistance(r1, c1, r2, c2) {
    const dx = Math.abs(r1 - r2);
    const dy = Math.abs(c1 - c2);
    const D = 1;
    const D2 = Math.sqrt(2);
    return D * (dx + dy) + (D2 - 2 * D) * Math.min(dx, dy);
}

function normalizeDistance(dist) {
    const maxDist = Math.sqrt(2) * (GRID_SIZE - 1);
    return dist / maxDist;
}

async function testModel() {
    try {
        console.log('Loading model...');
        const model = await tf.loadGraphModel('file://./web_model_static/model.json');
        console.log('✓ Model loaded');
        console.log('Inputs:', model.inputNames);
        console.log('Outputs:', model.outputNames);

        // Test Grid Variations (50 random grids, same start/goal)
        console.log('\n--- Testing Grid Variations ---');
        const startRow = 14, startCol = 3, goalRow = 7, goalCol = 13;
        const dist = octileDistance(startRow, startCol, goalRow, goalCol);
        const sgArray = [startRow/19, startCol/19, goalRow/19, goalCol/19, normalizeDistance(dist)];
        
        const predictions = [];
        for (let i = 0; i < 50; i++) {
            const gridArr = new Float32Array(400);
            let wallCount = 0;
            for (let j = 0; j < 400; j++) {
                if (Math.random() < 0.2) {
                    gridArr[j] = 1.0;
                    wallCount++;
                }
            }
            
            const gridTensor = tf.tensor4d(gridArr, [1, 20, 20, 1]);
            const sgTensor = tf.tensor2d([sgArray], [1, 5]);
            const out = model.execute({ grid: gridTensor, start_goal: sgTensor });
            const pred = (await out.data())[0];
            predictions.push({ test: i + 1, wallCount, pred });
            out.dispose();
            gridTensor.dispose();
            sgTensor.dispose();
        }
        
        const vals = predictions.map(p => p.pred);
        const minVal = Math.min(...vals);
        const maxVal = Math.max(...vals);
        const range = maxVal - minVal;
        console.log('Grid Variations (50 samples):');
        predictions.slice(0, 10).forEach(p => {
            console.log(`  Test ${p.test}: ${p.wallCount} walls -> ${p.pred.toFixed(6)}`);
        });
        console.log(`Min: ${minVal.toFixed(6)}, Max: ${maxVal.toFixed(6)}, Range: ${range.toFixed(6)}`);

        // Test Start/Goal Variations (50 pairs, same grid)
        console.log('\n--- Testing Start/Goal Variations ---');
        const gridArr = new Float32Array(400);
        for (let i = 0; i < 400; i++) {
            gridArr[i] = Math.random() < 0.2 ? 1.0 : 0.0;
        }
        
        const sgPredictions = [];
        for (let i = 0; i < 50; i++) {
            const sr = Math.floor(Math.random() * 20);
            const sc = Math.floor(Math.random() * 20);
            const gr = Math.floor(Math.random() * 20);
            const gc = Math.floor(Math.random() * 20);
            const d = octileDistance(sr, sc, gr, gc);
            const sg = [sr/19, sc/19, gr/19, gc/19, normalizeDistance(d)];
            
            const gridTensor = tf.tensor4d(gridArr, [1, 20, 20, 1]);
            const sgTensor = tf.tensor2d([sg], [1, 5]);
            const out = model.execute({ grid: gridTensor, start_goal: sgTensor });
            const pred = (await out.data())[0];
            sgPredictions.push({ test: i + 1, dist: d.toFixed(2), pred });
            out.dispose();
            gridTensor.dispose();
            sgTensor.dispose();
        }
        
        const sgVals = sgPredictions.map(p => p.pred);
        const sgMin = Math.min(...sgVals);
        const sgMax = Math.max(...sgVals);
        const sgRange = sgMax - sgMin;
        console.log('Start/Goal Variations (50 samples):');
        sgPredictions.slice(0, 10).forEach(p => {
            console.log(`  Test ${p.test}: dist=${p.dist} -> ${p.pred.toFixed(6)}`);
        });
        console.log(`Min: ${sgMin.toFixed(6)}, Max: ${sgMax.toFixed(6)}, Range: ${sgRange.toFixed(6)}`);
        
        if (sgRange > 0.05) {
            console.log('\n✓✓✓ SUCCESS! Start/Goal sensitivity is strong (range > 0.05)');
        } else if (sgRange > 0.01) {
            console.log('\n⚠ WARNING: Start/Goal sensitivity is weak (range only 0.01-0.05)');
        } else {
            console.log('\n✗ FAILURE: Start/Goal sensitivity is very weak (range < 0.01)');
        }

    } catch (e) {
        console.error('Error:', e.message);
    }
}

testModel();
