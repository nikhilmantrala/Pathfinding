import { Pathfinder } from './pathfinder.js';

export class BatchTestManager {
    constructor(pathfindingManager, uiManager, gridGenerator) {
        this.pathfindingManager = pathfindingManager;
        this.uiManager = uiManager;
        this.gridGenerator = gridGenerator;
        this.running = false;
        this.columnHeaders = [
            'Test Case',
            'Layout Type',
            'Wall Density',
            'Algorithm',
            'Time (ms)',
            'Nodes Visited',
            'Path Length',
            'Success'
        ];
        this.testResults = [this.columnHeaders];
        
        // Set up export button
        const exportBtn = this.uiManager?.elements?.batchTest?.exportBtn;
        if (exportBtn) {
            exportBtn.disabled = true;
            exportBtn.addEventListener('click', () => this.exportToExcel());
        }
    }

    runTests = async (config) => {
        if (!config || typeof config !== 'object') {
            throw new Error('Invalid configuration provided to runTests');
        }

        if (this.running) return;
        this.running = true;

        const stats = {
            totalNodes: 0,
            totalDistance: 0,
            totalTime: 0,
            runs: 0,
            successful: 0
        };

        try {
            // Show progress bar
            const progressBar = this.uiManager?.elements?.batchTest?.progress;
            const progressFill = this.uiManager?.elements?.batchTest?.progressBar;
            const progressText = this.uiManager?.elements?.batchTest?.progressText;

            if (progressBar && progressFill && progressText) {
                progressBar.style.display = 'block';
                progressFill.style.width = '0%';
                progressText.textContent = '0%';
            }

            this.uiManager?.disableControls?.();
            
            // Setup results table
            const table = document.createElement('table');
            const tbody = document.createElement('tbody');
            table.appendChild(tbody);
            
            if (this.uiManager?.elements?.results) {
                this.uiManager.elements.results.innerHTML = '';
                this.uiManager.elements.results.appendChild(table);
            }

            // Add header row
            const headerRow = tbody.insertRow();
            this.columnHeaders.forEach(text => {
                const th = document.createElement('th');
                th.textContent = text;
                headerRow.appendChild(th);
            });
            
            // Clear and initialize results array with headers
            this.testResults = [];
            this.testResults.push(this.columnHeaders);

            // Get algorithms to test
            const algorithms = config.algorithmPair?.algorithms || [];
            if (algorithms.length === 0) {
                throw new Error('No algorithms selected for testing');
            }

            const layouts = config.layouts || ['Random'];
            const densities = config.densities || ['Medium'];

            const totalRuns = (config.seedsPerFamily || 1) * 
                            (config.pairsPerSeed || 1) * 
                            algorithms.length * 
                            layouts.length * 
                            densities.length;
            let currentRun = 0;
            let testCaseId = 0;
            
            for (const layout of layouts) {
                for (const density of densities) {
                    for (let seed = 0; seed < (config.seedsPerFamily || 1); seed++) {
                        for (let pair = 0; pair < (config.pairsPerSeed || 1); pair++) {
                            testCaseId++;
                            // Generate a layout once and store its configuration
                            const gridConfig = await this.generateAndStoreLayout(layout, density);
                            
                            // Add a header row for this test case
                            const testCaseRow = tbody.insertRow();
                            testCaseRow.style.backgroundColor = '#f0f0f0';
                            testCaseRow.style.fontWeight = 'bold';
                            const testCaseCell = testCaseRow.insertCell();
                            testCaseCell.textContent = `Test Case #${testCaseId}`;
                            testCaseCell.colSpan = 8;
                            
                            for (const algorithm of algorithms) {
                                currentRun++;
                                // Use the same layout for all algorithms in this test case
                                const result = await this.runSingleTestWithLayout(
                                    algorithm,
                                    config.timeoutMs,
                                    gridConfig
                                );
                                
                                if (result.success) {
                                    stats.successful++;
                                    stats.totalNodes += result.nodes;
                                    stats.totalDistance += result.distance;
                                    stats.totalTime += result.time;
                                }
                                stats.runs++;

                                // Calculate wall percentage
                                const totalCells = this.gridGenerator.grid.rows * this.gridGenerator.grid.cols;
                                const wallCount = gridConfig.walls.length;
                                const wallPercentage = ((wallCount / totalCells) * 100).toFixed(1);
                                
                                // Create row data and add to results
                                const thisRowData = [
                                    testCaseId,
                                    layout,
                                    `${wallPercentage}%`,
                                    algorithm,
                                    result.time.toFixed(2),
                                    result.nodes || 0,
                                    result.success ? result.distance.toFixed(2) : '0',
                                    result.success ? 'Yes' : 'No'
                                ];
                                
                                // Store for export
                                this.testResults.push(thisRowData);

                                // Add row to table
                                const resultRow = tbody.insertRow();
                                thisRowData.forEach((value, index) => {
                                    const cell = resultRow.insertCell();
                                    cell.textContent = value;
                                    if (index === 0) cell.style.paddingLeft = '20px';
                                    if (index === 7) cell.style.color = result.success ? 'green' : 'red';
                                });

                                // Update progress
                                if (progressFill && progressText) {
                                    const progress = (currentRun / totalRuns) * 100;
                                    progressFill.style.width = `${progress}%`;
                                    progressText.textContent = `${Math.round(progress)}%`;
                                }

                                // Allow UI updates
                                await new Promise(resolve => setTimeout(resolve, 0));
                            }
                        }
                    }
                }
            }

            // Add summary row
            if (stats.successful > 0) {
                const avgRow = tbody.insertRow();
                avgRow.style.fontWeight = 'bold';
                avgRow.insertCell().textContent = 'Average';
                avgRow.insertCell().textContent = '-';
                avgRow.insertCell().textContent = (stats.totalTime / stats.successful).toFixed(2);
                avgRow.insertCell().textContent = (stats.totalNodes / stats.successful).toFixed(2);
                avgRow.insertCell().textContent = (stats.totalDistance / stats.successful).toFixed(2);
                avgRow.insertCell().textContent = `${stats.successful}/${stats.runs}`;
            }
        } catch (error) {
            console.error('Batch test error:', error);
            throw error;
        } finally {
            this.uiManager?.enableControls?.();
            if (this.uiManager?.elements?.batchTest?.progress) {
                this.uiManager.elements.batchTest.progress.style.display = 'none';
            }
            // Enable export button after tests complete
            const updateExportButton = this.uiManager?.elements?.batchTest?.exportBtn;
            if (updateExportButton) {
                updateExportButton.disabled = false;
            }
            this.running = false;
        }
    }

    generateAndStoreLayout = async (layout, density) => {
        // Generate the layout
        this.gridGenerator.generate(layout, density);
        
        // Store the configuration (walls and start/end points)
        const gridConfig = {
            layout: layout,
            density: density,
            walls: [],
            // Store wall positions
            valid: []
        };

        // Store wall and valid position information
        for (let y = 0; y < this.gridGenerator.grid.rows; y++) {
            for (let x = 0; x < this.gridGenerator.grid.cols; x++) {
                if (this.gridGenerator.grid.grid[y][x].isWall) {
                    gridConfig.walls.push({x, y});
                } else {
                    gridConfig.valid.push({x, y});
                }
            }
        }

        return gridConfig;
    }

    runSingleTestWithLayout = async (algorithm, timeoutMs, gridConfig) => {
        const startTime = performance.now();
        
        try {
            // Clear the grid
            this.gridGenerator.clearGrid();
            
            // Restore walls from configuration
            for (const wall of gridConfig.walls) {
                this.gridGenerator.grid.grid[wall.y][wall.x].isWall = true;
            }
            
            // Find valid positions
            const validPositions = [];
            for (let y = 0; y < this.gridGenerator.grid.rows; y++) {
                for (let x = 0; x < this.gridGenerator.grid.cols; x++) {
                    if (!this.gridGenerator.grid.grid[y][x].isWall) {
                        validPositions.push(this.gridGenerator.grid.grid[y][x]);
                    }
                }
            }
            
            if (validPositions.length < 2) {
                return {
                    success: false,
                    nodes: 0,
                    distance: 0,
                    time: performance.now() - startTime
                };
            }

            // Select random start and end points
            const startIdx = Math.floor(Math.random() * validPositions.length);
            let endIdx;
            do {
                endIdx = Math.floor(Math.random() * validPositions.length);
            } while (endIdx === startIdx);

            const start = validPositions[startIdx];
            let end = validPositions[endIdx];  // Changed to let since we modify it later

            // Ensure the selected points are far enough apart (at least 25% of the grid size)
            const minDistance = Math.floor(Math.sqrt(Math.pow(this.gridGenerator.grid.rows, 2) + Math.pow(this.gridGenerator.grid.cols, 2)) * 0.25);
            const actualDistance = Math.sqrt(Math.pow(start.row - end.row, 2) + Math.pow(start.col - end.col, 2));
            
            if (actualDistance < minDistance) {
                // Try to find a better end point
                let attempts = 100; // Prevent infinite loop
                while (attempts > 0) {
                    const newEndIdx = Math.floor(Math.random() * validPositions.length);
                    if (newEndIdx !== startIdx) {
                        const newEnd = validPositions[newEndIdx];
                        const newDistance = Math.sqrt(Math.pow(start.row - newEnd.row, 2) + Math.pow(start.col - newEnd.col, 2));
                        if (newDistance >= minDistance) {
                            end = validPositions[newEndIdx];
                            break;
                        }
                    }
                    attempts--;
                }
            }

            // Create pathfinder with appropriate settings
            const heuristicFn = this.pathfindingManager.getSelectedAlgorithm(algorithm);
            if (!heuristicFn) {
                throw new Error(`Invalid algorithm: ${algorithm}`);
            }

            const pf = new Pathfinder(this.gridGenerator.grid, heuristicFn);
            
            // Run pathfinding with a timeout
            const result = await Promise.race([
                pf.findPath(start, end),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Timeout')), timeoutMs)
                )
            ]);
            
            return {
                success: result.success,
                nodes: result.nodesVisited,
                distance: result.success ? result.distance : 0,
                time: performance.now() - startTime
            };
        } catch (error) {
            console.error('Error in single test:', error);
            return {
                success: false,
                nodes: 0,
                distance: 0,
                time: performance.now() - startTime
            };
        }
    }

    exportToExcel = () => {
        if (!this.testResults || this.testResults.length === 0) {
            console.warn('No test results to export');
            return;
        }

        try {
            // Create CSV content with UTF-8 BOM
            let csv = '\ufeff';
            
            // Add headers
            csv += this.columnHeaders.join(',') + '\n';
            
            // Add test results (skip header row from testResults)
            this.testResults.slice(1).forEach(row => {
                csv += row.join(',') + '\n';
            });

            // Create and trigger download
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            const now = new Date();
            const fileName = `pathfinding_results_${now.toISOString().split('T')[0]}.csv`;
            
            link.href = url;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            console.log('Export successful:', fileName);
        } catch (error) {
            console.error('Error exporting results:', error);
            alert('Failed to export results. Check console for details.');
        }
    }

    calculateSummaryStats = () => {
        // Skip header row
        const data = this.testResults.slice(1);
        
        const summary = {
            'Total Test Cases': new Set(data.map(row => row[0])).size,
            'Total Runs': data.length,
            'Successful Paths': data.filter(row => row[7] === 'Yes').length,
            'Average Time (ms)': (data.reduce((sum, row) => sum + parseFloat(row[4]), 0) / data.length).toFixed(2),
            'Average Nodes Visited': Math.round(data.reduce((sum, row) => sum + parseInt(row[5]), 0) / data.length),
            'Average Path Length': (data.reduce((sum, row) => sum + (row[7] === 'Yes' ? parseFloat(row[6]) : 0), 0) / 
                                  data.filter(row => row[7] === 'Yes').length).toFixed(2)
        };

        // Calculate success rate per algorithm
        const algorithmStats = {};
        data.forEach(row => {
            const algo = row[3]; // Algorithm is in column 4 (index 3)
            if (!algorithmStats[algo]) {
                algorithmStats[algo] = { total: 0, success: 0 };
            }
            algorithmStats[algo].total++;
            if (row[7] === 'Yes') algorithmStats[algo].success++;
        });

        Object.entries(algorithmStats).forEach(([algo, stats]) => {
            summary[`${algo} Success Rate`] = `${((stats.success / stats.total) * 100).toFixed(1)}%`;
        });

        return summary;
    }
}
