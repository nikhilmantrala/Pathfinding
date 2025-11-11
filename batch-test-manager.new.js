import { Pathfinder } from './pathfinder.js';

export class BatchTestManager {
    constructor(pathfindingManager, uiManager, gridGenerator) {
        this.pathfindingManager = pathfindingManager;
        this.uiManager = uiManager;
        this.gridGenerator = gridGenerator;
        this.running = false;
    }

    async runTests(config) {
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
            ['Run #', 'Algorithm', 'Time (ms)', 'Nodes', 'Distance', 'Success'].forEach(text => {
                const th = document.createElement('th');
                th.textContent = text;
                headerRow.appendChild(th);
            });

            // Get algorithms to test
            const algorithms = config.algorithmPair?.algorithms || [];
            if (algorithms.length === 0) {
                throw new Error('No algorithms selected for testing');
            }

            const totalRuns = (config.seedsPerFamily || 1) * (config.pairsPerSeed || 1) * algorithms.length;
            let currentRun = 0;

            for (let seed = 0; seed < (config.seedsPerFamily || 1); seed++) {
                for (let pair = 0; pair < (config.pairsPerSeed || 1); pair++) {
                    for (const algorithm of algorithms) {
                        currentRun++;
                        const result = await this.runSingleTest(
                            seed * config.pairsPerSeed + pair,
                            algorithm,
                            config.timeoutMs
                        );
                        
                        if (result.success) {
                            stats.successful++;
                            stats.totalNodes += result.nodes;
                            stats.totalDistance += result.distance;
                            stats.totalTime += result.time;
                        }
                        stats.runs++;

                        // Add result row
                        const row = tbody.insertRow();
                        row.insertCell().textContent = currentRun;
                        row.insertCell().textContent = algorithm;
                        row.insertCell().textContent = result.time.toFixed(2);
                        row.insertCell().textContent = result.nodes;
                        row.insertCell().textContent = result.success ? result.distance.toFixed(2) : 'No path';
                        row.insertCell().textContent = result.success ? '✓' : '✗';

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
            this.running = false;
        }
    }

    async runSingleTest(testNumber, algorithm, timeoutMs = 10000) {
        const startTime = performance.now();
        
        try {
            // Generate new layout
            this.gridGenerator.generate();
            
            // Find valid positions
            const validPositions = [];
            for (let y = 0; y < this.gridGenerator.ROWS; y++) {
                for (let x = 0; x < this.gridGenerator.COLS; x++) {
                    if (!this.gridGenerator.grid[y][x].isWall) {
                        validPositions.push({y, x});
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
            const end = validPositions[endIdx];

            // Run pathfinding
            const pf = new Pathfinder(this.gridGenerator.grid, algorithm);
            
            const result = await Promise.race([
                pf.findPath(start, end),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Timeout')), timeoutMs)
                )
            ]);

            return {
                success: result.path !== null,
                nodes: result.visitedNodes,
                distance: result.distance || 0,
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
}
