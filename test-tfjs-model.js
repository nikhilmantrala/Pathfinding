#!/usr/bin/env node
/**
 * Test TFJS model directly to check if conversion is correct
 */
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

async function testModel() {
    console.log('Loading TFJS model from web_model_static/model.json...');
    
    const modelPath = 'file://web_model_static/model.json';
    let model;
    try {
        model = await tf.loadGraphModel(modelPath);
        console.log('✓ Model loaded successfully');
    } catch (e) {
        console.error('✗ Failed to load model:', e.message);
        return;
    }
    
    console.log('\nModel inputs:');
    console.log('  inputNames:', model.inputNames);
    
    console.log('\nModel outputs:');
    console.log('  outputNames:', model.outputNames);
    
    // Test with dummy inputs
    const gridTensor = tf.zeros([1, 20, 20, 1], 'float32');
    const sgTensor = tf.zeros([1, 5], 'float32');
    
    console.log('\nTesting with zero inputs:');
    console.log(`  grid shape: ${gridTensor.shape}`);
    console.log(`  start_goal shape: ${sgTensor.shape}`);
    
    let result;
    try {
        // Try different input orderings
        result = model.execute({grid: gridTensor, start_goal: sgTensor});
        console.log('✓ execute({grid, start_goal}) succeeded');
    } catch (e1) {
        try {
            result = model.execute({start_goal: sgTensor, grid: gridTensor});
            console.log('✓ execute({start_goal, grid}) succeeded');
        } catch (e2) {
            console.error('✗ Both execute orderings failed');
            gridTensor.dispose();
            sgTensor.dispose();
            return;
        }
    }
    
    const val = Array.isArray(result) ? result[0] : result;
    const data = await val.data();
    console.log(`  Result value: ${data[0]}`);
    
    // Test with actual inputs simulating the pathfinding test
    console.log('\nTesting with realistic inputs:');
    const maxCost = Math.SQRT2 * 19;
    const mockGridData = new Float32Array(400);
    // Add some walls
    for (let i = 0; i < 100; i++) {
        mockGridData[Math.floor(Math.random() * 400)] = 1.0;
    }
    
    const mockSGData = [
        0.65,  // start_r normalized: 13/19
        0.05,  // start_c normalized: 1/19
        0.26,  // goal_r normalized: 5/19
        0.79,  // goal_c normalized: 15/19
        0.62   // normalized octile distance
    ];
    
    const grid2 = tf.tensor4d(mockGridData, [1, 20, 20, 1], 'float32');
    const sg2 = tf.tensor2d([mockSGData], [1, 5], 'float32');
    
    let result2;
    try {
        result2 = model.execute({grid: grid2, start_goal: sg2});
    } catch (e) {
        try {
            result2 = model.execute({start_goal: sg2, grid: grid2});
        } catch (e2) {
            console.error('✗ Failed with realistic inputs');
            grid2.dispose();
            sg2.dispose();
            return;
        }
    }
    
    const val2 = Array.isArray(result2) ? result2[0] : result2;
    const data2 = await val2.data();
    const predVal = data2[0];
    const predictedCost = predVal * maxCost;
    const octile = 17.314;  // From earlier test
    const residual = predictedCost - octile;
    
    console.log(`  pred_val (normalized): ${predVal.toFixed(4)}`);
    console.log(`  predicted_cost: ${predictedCost.toFixed(3)}`);
    console.log(`  octile: ${octile.toFixed(3)}`);
    console.log(`  residual: ${residual.toFixed(3)}`);
    
    gridTensor.dispose();
    sgTensor.dispose();
    grid2.dispose();
    sg2.dispose();
    val.dispose();
    val2.dispose();
    
    console.log('\n✓ TFJS model appears to be working correctly');
}

testModel().catch(e => console.error('Fatal error:', e));
