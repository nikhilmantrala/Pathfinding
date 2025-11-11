// Batch test handler function
async function handleBatchTest() {
    let testRunner;
    let controls;
    let progressSection;

    try {
        // Get and validate form inputs
        const seedPreset = document.getElementById('seedPreset');
        const pairsPerSeedInput = document.getElementById('pairsPerSeed');
        const timeoutInput = document.getElementById('timeoutMs');
        progressSection = document.getElementById('batchTestProgress');
        const resultsBody = document.getElementById('resultsBody');

        if (!seedPreset) throw new Error('Seeds per family input not found');
        if (!pairsPerSeedInput) throw new Error('Pairs per seed input not found');
        if (!timeoutInput) throw new Error('Timeout input not found');
        if (!progressSection) throw new Error('Progress section not found');
        if (!resultsBody) throw new Error('Results table not found');

        // Get and validate numeric inputs
        const seedsPerFamily = parseInt(seedPreset.value);
        const pairsPerSeed = parseInt(pairsPerSeedInput.value);
        const timeoutMs = parseInt(timeoutInput.value);

        if (isNaN(seedsPerFamily) || seedsPerFamily <= 0) {
            throw new Error('Please enter a valid number of seeds per family (must be greater than 0)');
        }
        if (isNaN(pairsPerSeed) || pairsPerSeed <= 0) {
            throw new Error('Please enter a valid number of pairs per seed (must be greater than 0)');
        }
        if (isNaN(timeoutMs) || timeoutMs <= 0) {
            throw new Error('Please enter a valid timeout value (must be greater than 0)');
        }

        // Get selections
        const selectedLayouts = Array.from(document.querySelectorAll('#layoutFamilies input:checked'))
            .map(cb => cb.value);
        const selectedDensities = Array.from(document.querySelectorAll('#densityBands input:checked'))
            .map(cb => cb.value);

        if (selectedLayouts.length === 0 || selectedDensities.length === 0) {
            throw new Error('Please select at least one layout family and density band');
        }

        // Create configuration
        const config = new BatchTestConfig();
        config.selectedLayouts = new Set(selectedLayouts);
        config.selectedDensities = new Set(selectedDensities);
        config.seedsPerFamily = seedsPerFamily;
        config.pairsPerSeed = pairsPerSeed;
        config.timeoutMs = timeoutMs;
        config.saveInputs = document.getElementById('saveInputs')?.checked || false;

        // Initialize progress elements
        const progressBar = progressSection.querySelector('.progress-fill');
        const progressText = progressSection.querySelector('.progress-text');
        if (!progressBar || !progressText) throw new Error('Progress elements not found');

        progressSection.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = '0%';

        // Store and disable controls
        controls = document.querySelectorAll('.batch-test-panel button, .batch-test-panel input, .batch-test-panel select');
        if (!controls || controls.length === 0) {
            throw new Error('No batch test controls found');
        }
        controls.forEach(control => control.disabled = true);

        // Create test instances
        const testGrid = Array(ROWS).fill().map((_, row) => 
            Array(COLS).fill().map((_, col) => new Cell(row, col))
        );

        const pathfinder = new Pathfinder({ 
            grid: testGrid, 
            rows: ROWS, 
            cols: COLS 
        }, getSelectedAlgorithm());

        // Create and configure test runner
        testRunner = new BatchTestRunner(config, pathfinder, gridGenerator);
        testRunner.onProgress = (progress) => {
            progressBar.style.width = `${progress * 100}%`;
            progressText.textContent = `${Math.round(progress * 100)}%`;
        };

        // Run tests
        console.log('Starting batch test run...');
        await testRunner.run();

        // Display results
        resultsBody.innerHTML = '';
        
        if (testRunner.results?.length > 0) {
            testRunner.results.forEach((result, index) => {
                const row = resultsBody.insertRow();
                [
                    index + 1,
                    result.layout_family || 'Unknown',
                    result.density_band || 'Unknown',
                    `${result.algorithm || 'Unknown'} (${result.pair_id || 0})`,
                    result.success ? '✓' : '✗',
                    result.path_cost != null ? result.path_cost.toFixed(2) : 'N/A',
                    result.nodes_expanded || 0,
                    result.runtime_ms != null ? result.runtime_ms.toFixed(2) : 'N/A'
                ].forEach(text => {
                    const cell = row.insertCell();
                    cell.textContent = text;
                });
            });

            const exportBtn = document.getElementById('exportCsvBtn');
            if (exportBtn) {
                exportBtn.onclick = () => {
                    try {
                        if (!testRunner.results?.length) {
                            throw new Error('No results to export');
                        }
                        const csv = testRunner.exportToCsv();
                        const blob = new Blob([csv], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `pathfinding_results_${new Date().toISOString().slice(0,10)}.csv`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                    } catch (error) {
                        console.error('Export error:', error);
                        alert('Failed to export results: ' + error.message);
                    }
                };
            }
        } else {
            const row = resultsBody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 8;
            cell.textContent = 'No results available';
            cell.style.textAlign = 'center';
            cell.style.padding = '10px';
        }
    } catch (error) {
        console.error('Batch test error:', error);
        alert('An error occurred during batch testing: ' + error.message);
    } finally {
        if (controls) {
            controls.forEach(control => {
                if (control) control.disabled = false;
            });
        }
        
        if (progressSection) {
            progressSection.style.display = 'none';
        }
        
        try {
            grid = Array(ROWS).fill().map((_, row) => 
                Array(COLS).fill().map((_, col) => new Cell(row, col))
            );
            start = null;
            end = null;
            gridGenerator = new GridGenerator(grid, ROWS, COLS);
            drawGrid();
        } catch (error) {
            console.error('Error resetting grid:', error);
        }
    }
}
