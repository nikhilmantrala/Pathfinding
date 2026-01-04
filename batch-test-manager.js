import { Pathfinder } from './pathfinder.js';

export class BatchTestManager {
    constructor(pathfindingManager, uiManager, gridGenerator) {
        this.pathfindingManager = pathfindingManager;
        this.uiManager = uiManager;
        this.gridGenerator = gridGenerator;
        this.running = false;
        this.obstacleEnvironmentMode = 'static';
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
        this.obstacleResults = [];
        this.obstacleSummaryStats = null;
        this.obstacleMlWins = [];
        
        const exportBtn = this.uiManager?.elements?.batchTest?.exportBtn;
        if (exportBtn) {
            exportBtn.disabled = true;
            exportBtn.addEventListener('click', () => this.exportToExcel());
        }
        
        const exportObstacleBtn = document.getElementById('exportObstacleResultsBtn');
        if (exportObstacleBtn) {
            exportObstacleBtn.disabled = true;
            exportObstacleBtn.addEventListener('click', () => this.exportObstacleAnalysis());
        }
        
        this.setupEnvironmentModeToggle();
    }

    setupEnvironmentModeToggle() {
        const staticBtn = document.getElementById('envModeStatic');
        const variableCostBtn = document.getElementById('envModeVariableCost');
        
        if (staticBtn && variableCostBtn) {
            staticBtn.addEventListener('click', () => {
                this.obstacleEnvironmentMode = 'static';
                staticBtn.classList.add('active');
                variableCostBtn.classList.remove('active');
            });
            
            variableCostBtn.addEventListener('click', () => {
                this.obstacleEnvironmentMode = 'variable_cost';
                variableCostBtn.classList.add('active');
                staticBtn.classList.remove('active');
            });
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
            const progressBar = this.uiManager?.elements?.batchTest?.progress;
            const progressFill = this.uiManager?.elements?.batchTest?.progressBar;
            const progressText = this.uiManager?.elements?.batchTest?.progressText;

            if (progressBar && progressFill && progressText) {
                progressBar.style.display = 'block';
                progressFill.style.width = '0%';
                progressText.textContent = '0%';
            }

            this.uiManager?.disableControls?.();
            
            const resultsDiv = document.getElementById('results');
            const tbody = document.getElementById('resultsBody');
            
            if (resultsDiv && tbody) {
                resultsDiv.style.display = 'block';
                tbody.innerHTML = '';
            }
            
            this.testResults = [];
            this.testResults.push(this.columnHeaders);

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
                            const gridConfig = await this.generateAndStoreLayout(layout, density);
                            
                            const testCaseRow = tbody.insertRow();
                            testCaseRow.style.backgroundColor = '#f0f0f0';
                            testCaseRow.style.fontWeight = 'bold';
                            const testCaseCell = testCaseRow.insertCell();
                            testCaseCell.textContent = `Test Case #${testCaseId}`;
                            testCaseCell.colSpan = 8;
                            
                            for (const algorithm of algorithms) {
                                currentRun++;
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

                                const totalCells = this.gridGenerator.grid.rows * this.gridGenerator.grid.cols;
                                const wallCount = gridConfig.walls.length;
                                const wallPercentage = ((wallCount / totalCells) * 100).toFixed(1);
                                
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
                                
                                this.testResults.push(thisRowData);

                                const resultRow = tbody.insertRow();
                                thisRowData.forEach((value, index) => {
                                    const cell = resultRow.insertCell();
                                    cell.textContent = value;
                                    if (index === 0) cell.style.paddingLeft = '20px';
                                    if (index === 7) cell.style.color = result.success ? 'green' : 'red';
                                });

                                if (progressFill && progressText) {
                                    const progress = (currentRun / totalRuns) * 100;
                                    progressFill.style.width = `${progress}%`;
                                    progressText.textContent = `${Math.round(progress)}%`;
                                }

                                await new Promise(resolve => setTimeout(resolve, 0));
                            }
                        }
                    }
                }
            }

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
            
            this.displayComparisonSummary(algorithms);
        } catch (error) {
            console.error('Batch test error:', error);
            throw error;
        } finally {
            this.uiManager?.enableControls?.();
            if (this.uiManager?.elements?.batchTest?.progress) {
                this.uiManager.elements.batchTest.progress.style.display = 'none';
            }
            const updateExportButton = this.uiManager?.elements?.batchTest?.exportBtn;
            if (updateExportButton) {
                updateExportButton.disabled = false;
            }
            this.running = false;
        }
    }

    generateAndStoreLayout = async (layout, density) => {
        this.gridGenerator.generate(layout, density);
        
        const gridConfig = {
            layout: layout,
            density: density,
            walls: [],
            valid: []
        };

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
            this.gridGenerator.clearGrid();
            
            for (const wall of gridConfig.walls) {
                this.gridGenerator.grid.grid[wall.y][wall.x].isWall = true;
            }
            
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

            const startIdx = Math.floor(Math.random() * validPositions.length);
            let endIdx;
            do {
                endIdx = Math.floor(Math.random() * validPositions.length);
            } while (endIdx === startIdx);

            const start = validPositions[startIdx];
            let end = validPositions[endIdx];

            const minDistance = Math.floor(Math.sqrt(Math.pow(this.gridGenerator.grid.rows, 2) + Math.pow(this.gridGenerator.grid.cols, 2)) * 0.25);
            const actualDistance = Math.sqrt(Math.pow(start.row - end.row, 2) + Math.pow(start.col - end.col, 2));
            
            if (actualDistance < minDistance) {
                let attempts = 100;
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

            const heuristicFn = this.pathfindingManager.getSelectedAlgorithm(algorithm);
            if (!heuristicFn) {
                throw new Error(`Invalid algorithm: ${algorithm}`);
            }

            const pf = new Pathfinder(this.gridGenerator.grid, heuristicFn);
            
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
            let csv = '\ufeff';
            
            csv += this.columnHeaders.join(',') + '\n';
            
            this.testResults.slice(1).forEach(row => {
                csv += row.join(',') + '\n';
            });

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
        } catch (error) {
            console.error('Error exporting results:', error);
            alert('Failed to export results. Check console for details.');
        }
    }

    exportObstacleAnalysis = async () => {
        if (!this.obstacleResults || this.obstacleResults.length === 0) {
            console.warn('No obstacle analysis results to export');
            alert('No obstacle analysis results available. Please run the analysis first.');
            return;
        }

        try {
            const now = new Date();
            
            // Calculate summary statistics
            let astarWins = 0, mlWins = 0, ties = 0;
            let totalAstarNodes = 0, totalMlNodes = 0;
            let totalAstarPath = 0, totalMlPath = 0;
            let totalAstarTime = 0, totalMlTime = 0;
            let successfulTests = 0;
            
            this.obstacleResults.forEach(result => {
                if (result.winner === 'astar') astarWins++;
                if (result.winner === 'ml') mlWins++;
                if (result.winner === 'tie') ties++;
                
                if (result.astar.success && result.ml.success) {
                    totalAstarNodes += result.astar.nodes;
                    totalMlNodes += result.ml.nodes;
                    totalAstarPath += result.astar.pathLength;
                    totalMlPath += result.ml.pathLength;
                    totalAstarTime += result.astar.time;
                    totalMlTime += result.ml.time;
                    successfulTests++;
                }
            });
            
            // Create workbook
            const wb = XLSX.utils.book_new();
            
            // ========== SHEET 1: SUMMARY STATISTICS ==========
            const summaryData = [
                ['Obstacle Analysis Report'],
                [`Generated: ${now.toISOString()}`],
                [`Total Tests: ${this.obstacleResults.length}`],
                [],
                ['Summary Statistics'],
                ['Metric', 'Value'],
                ['Total Tests', this.obstacleResults.length],
                ['A* Wins', astarWins],
                ['ML Wins', mlWins],
                ['Ties', ties],
                ['Successful Tests', successfulTests],
            ];
            
            if (successfulTests > 0) {
                summaryData.push(
                    [],
                    ['Average Performance'],
                    ['Metric', 'A*', 'ML', 'ML Improvement'],
                    ['Nodes Expanded', 
                        (totalAstarNodes / successfulTests).toFixed(2), 
                        (totalMlNodes / successfulTests).toFixed(2), 
                        `${(((totalAstarNodes - totalMlNodes) / totalAstarNodes) * 100).toFixed(1)}%`
                    ],
                    ['Path Length', 
                        (totalAstarPath / successfulTests).toFixed(2), 
                        (totalMlPath / successfulTests).toFixed(2), 
                        `${(((totalAstarPath - totalMlPath) / totalAstarPath) * 100).toFixed(1)}%`
                    ],
                    ['Time (ms)', 
                        (totalAstarTime / successfulTests).toFixed(2), 
                        (totalMlTime / successfulTests).toFixed(2), 
                        `${(((totalAstarTime - totalMlTime) / totalAstarTime) * 100).toFixed(1)}%`
                    ]
                );
            }
            
            // Add chart data for Summary
            summaryData.push(
                [],
                ['CHART DATA - Win Distribution'],
                ['Category', 'Count'],
                ['A* Wins', astarWins],
                ['ML Wins', mlWins],
                ['Ties', ties]
            );
            
            const ws1 = XLSX.utils.aoa_to_sheet(summaryData);
            XLSX.utils.book_append_sheet(wb, ws1, 'Summary Statistics');
            
            // Generate Win Distribution Chart
            const chartCanvas1 = await this.generateWinDistributionChart(astarWins, mlWins, ties);
            
            // ========== SHEET 2: ML WINS ==========
            const mlWinsData = [
                ['ML WINS - Cases Where ML Beat A*'],
                []
            ];
            
            if (this.obstacleMlWins && this.obstacleMlWins.length > 0) {
                mlWinsData.push(
                    ['Case #', 'Obstacle Count', 'A* Nodes', 'ML Nodes', 'Nodes Saved', 'Nodes Saved %', 'A* Path Length', 'ML Path Length', 'Path Improvement']
                );
                
                this.obstacleMlWins.forEach((win, index) => {
                    const nodesSaved = win.astar.nodes - win.ml.nodes;
                    const nodesSavedPct = ((nodesSaved / win.astar.nodes) * 100).toFixed(1);
                    const pathImprovement = ((win.astar.pathLength - win.ml.pathLength) / win.astar.pathLength * 100).toFixed(1);
                    
                    mlWinsData.push([
                        index + 1,
                        win.obstacleCount,
                        win.astar.nodes,
                        win.ml.nodes,
                        nodesSaved,
                        `${nodesSavedPct}%`,
                        win.astar.pathLength.toFixed(2),
                        win.ml.pathLength.toFixed(2),
                        `${pathImprovement}%`
                    ]);
                });
                
                // Add chart data
                mlWinsData.push(
                    [],
                    ['CHART DATA - Nodes Saved by Obstacle Count'],
                    ['Obstacle Count', 'Nodes Saved', 'Nodes Saved %']
                );
                
                this.obstacleMlWins.forEach(win => {
                    const nodesSaved = win.astar.nodes - win.ml.nodes;
                    const nodesSavedPct = ((nodesSaved / win.astar.nodes) * 100).toFixed(1);
                    mlWinsData.push([win.obstacleCount, nodesSaved, nodesSavedPct]);
                });
            } else {
                mlWinsData.push(['No cases found where ML beat A* in this test run.']);
            }
            
            const ws2 = XLSX.utils.aoa_to_sheet(mlWinsData);
            XLSX.utils.book_append_sheet(wb, ws2, 'ML Wins');
            
            // Generate ML Wins Chart
            const chartCanvas2 = this.obstacleMlWins.length > 0 ? 
                await this.generateNodesSavedChart(this.obstacleMlWins) : null;
            
            // ========== SHEET 3: DETAILED RESULTS ==========
            const detailedData = [
                ['DETAILED RESULTS - All Tests'],
                [],
                ['Test #', 'Obstacle Count', 'Winner', 'A* Nodes', 'ML Nodes', 'Nodes Saved', 'Nodes Saved %', 'A* Path Length', 'ML Path Length', 'A* Time (ms)', 'ML Time (ms)', 'A* Success', 'ML Success']
            ];
            
            this.obstacleResults.forEach((result, index) => {
                const nodesSaved = result.astar.nodes - result.ml.nodes;
                const nodesSavedPct = result.astar.nodes > 0 ? ((nodesSaved / result.astar.nodes) * 100).toFixed(1) : '0';
                
                detailedData.push([
                    index + 1,
                    result.obstacleCount,
                    result.winner,
                    result.astar.nodes,
                    result.ml.nodes,
                    nodesSaved,
                    `${nodesSavedPct}%`,
                    result.astar.pathLength.toFixed(2),
                    result.ml.pathLength.toFixed(2),
                    result.astar.time.toFixed(2),
                    result.ml.time.toFixed(2),
                    result.astar.success ? 'Yes' : 'No',
                    result.ml.success ? 'Yes' : 'No'
                ]);
            });
            
            // Add chart data
            detailedData.push(
                [],
                ['CHART DATA - Nodes by Obstacle Count'],
                ['Obstacle Count', 'A* Nodes', 'ML Nodes']
            );
            
            this.obstacleResults.forEach(result => {
                if (result.astar.success && result.ml.success) {
                    detailedData.push([result.obstacleCount, result.astar.nodes, result.ml.nodes]);
                }
            });
            
            detailedData.push(
                [],
                ['CHART DATA - Path Length by Obstacle Count'],
                ['Obstacle Count', 'A* Path Length', 'ML Path Length']
            );
            
            this.obstacleResults.forEach(result => {
                if (result.astar.success && result.ml.success) {
                    detailedData.push([
                        result.obstacleCount, 
                        parseFloat(result.astar.pathLength.toFixed(2)), 
                        parseFloat(result.ml.pathLength.toFixed(2))
                    ]);
                }
            });
            
            const ws3 = XLSX.utils.aoa_to_sheet(detailedData);
            XLSX.utils.book_append_sheet(wb, ws3, 'Detailed Results');
            
            // Generate comparison charts
            const chartCanvas3 = await this.generateNodesComparisonChart(this.obstacleResults);
            const chartCanvas4 = await this.generatePathComparisonChart(this.obstacleResults);
            
            // ========== SHEET 4: CHARTS ==========
            const chartsData = [
                ['CHARTS AND VISUALIZATIONS'],
                [],
                ['NOTE: Chart images are provided in the "Chart Images" sheet as Base64 data'],
                ['You can copy the Base64 data and use online converters to view images'],
                ['Some Excel versions may support clicking the hyperlinks in the Image Link column below'],
                [],
                ['Chart Name', 'Description', 'Image Link', 'Instructions'],
                ['1. Win Distribution', 'Shows A* vs ML wins across all density levels', 'Check below for hyperlink', 'Use "Chart Images" sheet for Base64 data'],
                [],
                [],
                [],
                [],
                ['2. Nodes Saved (ML Wins Only)', 'Shows node savings when ML outperforms A*', 'Check below for hyperlink', 'Use "Chart Images" sheet for Base64 data'],
                [],
                [],
                [],
                [],
                ['3. Nodes Comparison (All Tests)', 'Average nodes visited by algorithm and density', 'Check below for hyperlink', 'Use "Chart Images" sheet for Base64 data'],
                [],
                [],
                [],
                [],
                ['4. Path Length Comparison (All Tests)', 'Path length comparison across test scenarios', 'Check below for hyperlink', 'Use "Chart Images" sheet for Base64 data']
            ];
            
            const ws4 = XLSX.utils.aoa_to_sheet(chartsData);
            XLSX.utils.book_append_sheet(wb, ws4, 'Charts');
            
            // Add chart images to the Charts sheet
            const imageData = [];
            const downloadableImages = []; // For creating separate image files
            
            if (chartCanvas1) {
                const imageData1 = chartCanvas1.toDataURL('image/png');
                imageData.push(['Win Distribution', imageData1.split(',')[1]]); // Store only base64 part
                downloadableImages.push({ canvas: chartCanvas1, name: 'chart1_win_distribution' });
                this.addChartToSheet(wb, 'Charts', chartCanvas1, 8);
            }
            if (chartCanvas2) {
                const imageData2 = chartCanvas2.toDataURL('image/png');
                imageData.push(['Nodes Comparison', imageData2.split(',')[1]]);
                downloadableImages.push({ canvas: chartCanvas2, name: 'chart2_nodes_saved' });
                this.addChartToSheet(wb, 'Charts', chartCanvas2, 14);
            }
            if (chartCanvas3) {
                const imageData3 = chartCanvas3.toDataURL('image/png');
                imageData.push(['Nodes vs Obstacle Count', imageData3.split(',')[1]]);
                downloadableImages.push({ canvas: chartCanvas3, name: 'chart3_nodes_vs_obstacles' });
                this.addChartToSheet(wb, 'Charts', chartCanvas3, 20);
            }
            if (chartCanvas4) {
                const imageData4 = chartCanvas4.toDataURL('image/png');
                imageData.push(['Path Length vs Obstacle Count', imageData4.split(',')[1]]);
                downloadableImages.push({ canvas: chartCanvas4, name: 'chart4_path_vs_obstacles' });
                this.addChartToSheet(wb, 'Charts', chartCanvas4, 26);
            }
            
            // Create downloadable image files
            this.createDownloadableImages(downloadableImages);
            
            // Create a separate sheet with base64 image data for manual extraction
            if (imageData.length > 0) {
                const imageSheetData = [
                    ['Chart Images (Base64 Data)'],
                    ['Instructions: Base64 data is split across multiple cells due to Excel limitations'],
                    ['To reconstruct: Concatenate all "Part X" cells for each chart'],
                    ['Then use online Base64-to-image converter or HTML method below'],
                    [''],
                    ['HTML Method: Create a .html file with: <img src="data:image/png;base64,CONCATENATED_DATA_HERE">'],
                    ['']
                ];
                
                // Add each image with data split across multiple cells
                imageData.forEach((item, index) => {
                    const [chartName, base64Data] = item;
                    
                    // Add chart header
                    imageSheetData.push([`Chart ${index + 1}: ${chartName}`]);
                    imageSheetData.push(['Total Length:', `${base64Data.length} characters`]);
                    
                    // Split base64 data into chunks of 30,000 characters to stay under Excel's 32,767 limit
                    const chunkSize = 30000;
                    const chunks = [];
                    for (let i = 0; i < base64Data.length; i += chunkSize) {
                        chunks.push(base64Data.substring(i, i + chunkSize));
                    }
                    
                    // Add header for data chunks
                    imageSheetData.push(['Part', 'Base64 Data Chunk']);
                    
                    // Add each chunk as a separate row
                    chunks.forEach((chunk, chunkIndex) => {
                        imageSheetData.push([`Part ${chunkIndex + 1}`, chunk]);
                    });
                    
                    // Add empty rows between charts
                    imageSheetData.push(['']);
                    imageSheetData.push(['']);
                });
                
                const imageWs = XLSX.utils.aoa_to_sheet(imageSheetData);
                
                // Set column widths for better viewing
                imageWs['!cols'] = [
                    { width: 15 },  // Chart Name/Part column
                    { width: 120 }  // Base64 data column (very wide for long strings)
                ];
                
                XLSX.utils.book_append_sheet(wb, imageWs, 'Chart Images');
            }
            
            // Write file
            const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
            const blob = new Blob([wbout], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            const fileName = `obstacle_analysis_${now.toISOString().split('T')[0]}_${now.getHours()}-${now.getMinutes()}.xlsx`;
            
            link.href = url;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            alert(`Exported ${this.obstacleResults.length} test results with charts to ${fileName}\n\nNOTE: Chart images are included as Base64 data in the "Chart Images" sheet.\nYou can:\n1. Copy the Base64 data and use an online Base64-to-image converter\n2. Try clicking hyperlinks in the "Charts" sheet (may work in some Excel versions)\n3. Save Base64 data as .html file with <img src="data:image/png;base64,PASTE_DATA_HERE"> tag`);
        } catch (error) {
            console.error('Error exporting obstacle analysis:', error);
            alert('Failed to export obstacle analysis. Check console for details.');
        }
    }
    
    // Helper function to generate Win Distribution Chart
    generateWinDistributionChart = async (astarWins, mlWins, ties) => {
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 400;
        
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['A* Wins', 'ML Wins', 'Ties'],
                datasets: [{
                    label: 'Count',
                    data: [astarWins, mlWins, ties],
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(107, 114, 128, 0.7)'
                    ],
                    borderColor: [
                        'rgba(59, 130, 246, 1)',
                        'rgba(16, 185, 129, 1)',
                        'rgba(107, 114, 128, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Win Distribution - A* vs ML',
                        font: { size: 18, weight: 'bold' }
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Tests'
                        }
                    }
                }
            }
        });
        
        await new Promise(resolve => setTimeout(resolve, 500));
        return canvas;
    }
    
    // Helper function to generate Nodes Saved Chart
    generateNodesSavedChart = async (mlWins) => {
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 400;
        
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Nodes Saved',
                    data: mlWins.map(win => ({
                        x: win.obstacleCount,
                        y: win.astar.nodes - win.ml.nodes
                    })),
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Nodes Saved by ML (Cases Where ML Won)',
                        font: { size: 18, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Obstacle Count'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Nodes Saved'
                        }
                    }
                }
            }
        });
        
        await new Promise(resolve => setTimeout(resolve, 500));
        return canvas;
    }
    
    generateNodesComparisonChart = async (results) => {
        const canvas = document.createElement('canvas');
        canvas.width = 1000;
        canvas.height = 500;
        
        const successfulResults = results.filter(r => r.astar.success && r.ml.success);
        
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: successfulResults.map(r => r.obstacleCount),
                datasets: [
                    {
                        label: 'A* Nodes',
                        data: successfulResults.map(r => r.astar.nodes),
                        borderColor: 'rgba(59, 130, 246, 1)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: true
                    },
                    {
                        label: 'ML Nodes',
                        data: successfulResults.map(r => r.ml.nodes),
                        borderColor: 'rgba(16, 185, 129, 1)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Nodes Expanded: A* vs ML by Obstacle Count',
                        font: { size: 18, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Obstacle Count'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Nodes Expanded'
                        }
                    }
                }
            }
        });
        
        await new Promise(resolve => setTimeout(resolve, 500));
        return canvas;
    }
    
    generatePathComparisonChart = async (results) => {
        const canvas = document.createElement('canvas');
        canvas.width = 1000;
        canvas.height = 500;
        
        const successfulResults = results.filter(r => r.astar.success && r.ml.success);
        
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: successfulResults.map(r => r.obstacleCount),
                datasets: [
                    {
                        label: 'A* Path Length',
                        data: successfulResults.map(r => r.astar.pathLength),
                        borderColor: 'rgba(59, 130, 246, 1)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: true
                    },
                    {
                        label: 'ML Path Length',
                        data: successfulResults.map(r => r.ml.pathLength),
                        borderColor: 'rgba(16, 185, 129, 1)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Path Length: A* vs ML by Obstacle Count',
                        font: { size: 18, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Obstacle Count'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Path Length'
                        }
                    }
                }
            }
        });
        
        await new Promise(resolve => setTimeout(resolve, 500));
        return canvas;
    }
    
    addChartToSheet = (workbook, sheetName, canvas, startRow) => {
        try {
            const imageData = canvas.toDataURL('image/png');
            const base64 = imageData.split(',')[1];
            
            // Get the worksheet
            const worksheet = workbook.Sheets[sheetName];
            if (!worksheet) return;
            
            // Method 1: Add image reference as hyperlink (clickable in some Excel versions)
            const imageCell = XLSX.utils.encode_cell({ c: 2, r: startRow }); // Column C
            worksheet[imageCell] = {
                t: 's',
                v: '[Chart Image - Right-click → Open Hyperlink]',
                l: { Target: imageData, Tooltip: 'Click to view chart image' }
            };
            
            // Method 2: Add image dimensions and instructions
            const instructionCell = XLSX.utils.encode_cell({ c: 3, r: startRow });
            worksheet[instructionCell] = {
                t: 's',
                v: 'Chart available in "Chart Images" sheet as Base64 data'
            };
            
            // Method 3: Try to add as Excel drawing object (may work in some versions)
            try {
                if (!worksheet['!drawing']) {
                    worksheet['!drawing'] = [];
                }
                
                worksheet['!drawing'].push({
                    type: 'image',
                    data: base64,
                    format: 'png',
                    position: {
                        from: { col: 2, row: startRow },
                        to: { col: 7, row: startRow + 4 }
                    }
                });
            } catch (drawingError) {
                // Drawing method failed, that's okay
            }
            
        } catch (error) {
            console.warn('Could not add chart image to Excel:', error);
            // Fallback: Add a text reference to the chart
            const worksheet = workbook.Sheets[sheetName];
            if (worksheet) {
                const imageCell = XLSX.utils.encode_cell({ c: 2, r: startRow });
                worksheet[imageCell] = {
                    t: 's',
                    v: '[Chart image - check "Chart Images" sheet for Base64 data]'
                };
            }
        }
    }
    
    createDownloadableImages = (chartCanvases) => {
        chartCanvases.forEach((canvasInfo, index) => {
            try {
                const { canvas, name } = canvasInfo;
                const imageData = canvas.toDataURL('image/png');
                
                // Create downloadable link
                const link = document.createElement('a');
                link.download = `${name || `chart_${index + 1}`}.png`;
                link.href = imageData;
                link.style.display = 'none';
                
                // Add to DOM, click, and remove
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                // Small delay between downloads
                setTimeout(() => {}, 100 * index);
            } catch (error) {
                console.warn(`Could not create downloadable image for chart ${index + 1}:`, error);
            }
        });
    }

    calculateSummaryStats = () => {
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

        const algorithmStats = {};
        data.forEach(row => {
            const algo = row[3];
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

    displayComparisonSummary = (algorithms) => {
        const summaryDiv = document.getElementById('testSummary');
        const winnerDiv = document.getElementById('summaryWinner');
        const gridDiv = document.getElementById('summaryGrid');
        const verdictDiv = document.getElementById('summaryVerdict');
        
        if (!summaryDiv || !winnerDiv || !gridDiv || !verdictDiv) return;
        
        // Skip header row
        const data = this.testResults.slice(1);
        if (data.length === 0) return;
        
        // Calculate stats per algorithm
        const algoStats = {};
        algorithms.forEach(algo => {
            const algoData = data.filter(row => row[3] === algo);
            const successfulRuns = algoData.filter(row => row[7] === 'Yes');
            
            algoStats[algo] = {
                totalRuns: algoData.length,
                successCount: successfulRuns.length,
                successRate: algoData.length > 0 ? (successfulRuns.length / algoData.length) * 100 : 0,
                avgTime: successfulRuns.length > 0 
                    ? successfulRuns.reduce((sum, row) => sum + parseFloat(row[4]), 0) / successfulRuns.length 
                    : 0,
                avgNodes: successfulRuns.length > 0 
                    ? successfulRuns.reduce((sum, row) => sum + parseInt(row[5]), 0) / successfulRuns.length 
                    : 0,
                avgPathLength: successfulRuns.length > 0 
                    ? successfulRuns.reduce((sum, row) => sum + parseFloat(row[6]), 0) / successfulRuns.length 
                    : 0
            };
        });
        
        // Determine winner based on multiple criteria
        const algoNames = Object.keys(algoStats);
        if (algoNames.length < 2) {
            summaryDiv.style.display = 'none';
            return;
        }
        
        // Score each algorithm (lower is better for time and nodes, higher for success rate)
        const scores = {};
        algoNames.forEach(algo => scores[algo] = 0);
        
        // Compare metrics pairwise
        const comparisons = {
            time: { metric: 'avgTime', lowerIsBetter: true, weight: 2, label: 'Faster' },
            nodes: { metric: 'avgNodes', lowerIsBetter: true, weight: 1.5, label: 'More Efficient' },
            successRate: { metric: 'successRate', lowerIsBetter: false, weight: 2, label: 'More Reliable' },
            pathLength: { metric: 'avgPathLength', lowerIsBetter: true, weight: 1, label: 'Shorter Paths' }
        };
        
        const metricWinners = {};
        
        Object.entries(comparisons).forEach(([key, config]) => {
            let bestAlgo = algoNames[0];
            let bestValue = algoStats[algoNames[0]][config.metric];
            
            algoNames.slice(1).forEach(algo => {
                const value = algoStats[algo][config.metric];
                const isBetter = config.lowerIsBetter 
                    ? value < bestValue 
                    : value > bestValue;
                if (isBetter) {
                    bestAlgo = algo;
                    bestValue = value;
                }
            });
            
            scores[bestAlgo] += config.weight;
            metricWinners[key] = { algo: bestAlgo, label: config.label };
        });
        
        // Find overall winner
        let overallWinner = algoNames[0];
        let highestScore = scores[algoNames[0]];
        algoNames.slice(1).forEach(algo => {
            if (scores[algo] > highestScore) {
                overallWinner = algo;
                highestScore = scores[algo];
            }
        });
        
        // Check if it's a tie
        const isTie = algoNames.every(algo => scores[algo] === highestScore);
        
        // Build winner section
        const winnerIcon = isTie ? '🤝' : '🏆';
        const winnerText = isTie ? 'It\'s a Tie!' : overallWinner.toUpperCase();
        winnerDiv.innerHTML = `
            <div class="winner-label">${isTie ? 'Results are even!' : 'Best Performing Algorithm'}</div>
            <div class="winner-name"><span class="winner-icon">${winnerIcon}</span>${winnerText}</div>
        `;
        
        // Build comparison cards
        let cardsHtml = '';
        algoNames.forEach(algo => {
            const stats = algoStats[algo];
            const isWinner = algo === overallWinner && !isTie;
            
            cardsHtml += `
                <div class="summary-card" style="${isWinner ? 'border: 2px solid #4ade80;' : ''}">
                    <div class="algo-name">${algo.toUpperCase()} ${isWinner ? '👑' : ''}</div>
                    <div class="stat-row">
                        <span class="stat-label">Success Rate:</span>
                        <span class="stat-value ${this.getCompareClass(algo, 'successRate', algoStats, false)}">
                            ${stats.successRate.toFixed(1)}%
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Time:</span>
                        <span class="stat-value ${this.getCompareClass(algo, 'avgTime', algoStats, true)}">
                            ${stats.avgTime.toFixed(2)} ms
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Nodes:</span>
                        <span class="stat-value ${this.getCompareClass(algo, 'avgNodes', algoStats, true)}">
                            ${Math.round(stats.avgNodes)}
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Path Length:</span>
                        <span class="stat-value ${this.getCompareClass(algo, 'avgPathLength', algoStats, true)}">
                            ${stats.avgPathLength.toFixed(2)}
                        </span>
                    </div>
                </div>
            `;
        });
        gridDiv.innerHTML = cardsHtml;
        
        // Helper function to format algorithm names properly
        const formatAlgoName = (algo) => {
            if (algo.toLowerCase() === 'astar') return 'A*';
            if (algo.toLowerCase() === 'ml') return 'ML';
            return algo.toUpperCase();
        };
        
        // Build verdict section with comparison bar
        const algo1 = algoNames[0];
        const algo2 = algoNames[1];
        const stats1 = algoStats[algo1];
        const stats2 = algoStats[algo2];
        
        // Calculate win percentages for all metrics
        const timeDiff = ((stats2.avgTime - stats1.avgTime) / Math.max(stats1.avgTime, stats2.avgTime) * 100);
        const nodesDiff = ((stats2.avgNodes - stats1.avgNodes) / Math.max(stats1.avgNodes, stats2.avgNodes) * 100);
        const pathDiff = ((stats2.avgPathLength - stats1.avgPathLength) / Math.max(stats1.avgPathLength, stats2.avgPathLength) * 100);
        
        let verdictLines = [];
        
        // Time comparison
        if (Math.abs(timeDiff) > 1) {
            const faster = timeDiff > 0 ? algo1 : algo2;
            const pct = Math.abs(timeDiff).toFixed(1);
            verdictLines.push(`⚡ <strong>${formatAlgoName(faster)}</strong> is ${pct}% faster`);
        }
        
        // Nodes comparison
        if (Math.abs(nodesDiff) > 1) {
            const efficient = nodesDiff > 0 ? algo1 : algo2;
            const pct = Math.abs(nodesDiff).toFixed(1);
            verdictLines.push(`🔍 <strong>${formatAlgoName(efficient)}</strong> explores ${pct}% fewer nodes`);
        }
        
        // Path length comparison
        if (Math.abs(pathDiff) > 1) {
            const shorter = pathDiff > 0 ? algo1 : algo2;
            const pct = Math.abs(pathDiff).toFixed(1);
            verdictLines.push(`📏 <strong>${formatAlgoName(shorter)}</strong> finds ${pct}% shorter paths`);
        }
        
        // Success rate comparison
        const successDiff = stats1.successRate - stats2.successRate;
        if (Math.abs(successDiff) > 1) {
            const reliable = successDiff > 0 ? algo1 : algo2;
            const reliablePct = Math.abs(successDiff).toFixed(1);
            verdictLines.push(`✅ <strong>${formatAlgoName(reliable)}</strong> has ${reliablePct}% higher success rate`);
        }
        
        if (verdictLines.length === 0) {
            verdictLines.push('📊 Both algorithms performed similarly in this test');
        }
        
        verdictDiv.innerHTML = `
            <div class="verdict-title">Key Insights</div>
            <div class="verdict-text">${verdictLines.join('<br>')}</div>
        `;
        
        // Show the summary section
        summaryDiv.style.display = 'block';
    }
    
    getCompareClass = (algo, metric, algoStats, lowerIsBetter) => {
        const algoNames = Object.keys(algoStats);
        if (algoNames.length < 2) return '';
        
        const values = algoNames.map(a => ({ algo: a, value: algoStats[a][metric] }));
        values.sort((a, b) => lowerIsBetter ? a.value - b.value : b.value - a.value);
        
        if (values[0].algo === algo) return 'better';
        if (values[values.length - 1].algo === algo) return 'worse';
        return '';
    }

    runObstacleComparison = async () => {
        if (this.running) return;
        this.running = true;

        const resultsDiv = document.getElementById('obstacleTestResults');
        const progressDiv = document.getElementById('obstacleProgress');
        const mlWinsTable = document.getElementById('mlWinsTable');
        const summaryDiv = document.getElementById('obstacleSummary');
        const obstacleBtn = document.getElementById('obstacleTestBtn');

        if (!resultsDiv || !progressDiv) {
            console.error('Obstacle test elements not found');
            this.running = false;
            return;
        }

        if (obstacleBtn) obstacleBtn.disabled = true;

        resultsDiv.style.display = 'block';
        
        const isVariableCost = this.obstacleEnvironmentMode === 'variable_cost';
        const algo1 = isVariableCost ? 'astar_variable_cost' : 'astar';
        const algo2 = isVariableCost ? 'ml_variable_cost' : 'ml';
        const envLabel = isVariableCost ? 'Variable Cost Environment' : 'Static Cost Environment';
        
        progressDiv.innerHTML = `🚀 Starting ${algo1} vs ${algo2} Obstacle Comparison (${envLabel})...\n`;
        mlWinsTable.innerHTML = '';
        summaryDiv.innerHTML = '';

        const TOTAL_TESTS = parseInt(document.getElementById('obstacleTestCount')?.value) || 100;
        const MIN_OBSTACLES = parseInt(document.getElementById('obstacleMin')?.value) || 15;
        const MAX_OBSTACLES = parseInt(document.getElementById('obstacleMax')?.value) || 100;
        const TEST_TIMEOUT = parseInt(document.getElementById('obstacleTimeout')?.value) || 10000;

        const mlWins = [];
        const allResults = [];

        const obstacleCounts = [];
        for (let i = 0; i < TOTAL_TESTS; i++) {
            const obstacleCount = Math.round(MIN_OBSTACLES + (MAX_OBSTACLES - MIN_OBSTACLES) * (i / (TOTAL_TESTS - 1)));
            obstacleCounts.push(obstacleCount);
        }

        progressDiv.innerHTML += `📊 Configuration: ${TOTAL_TESTS} tests, ${MIN_OBSTACLES}-${MAX_OBSTACLES} obstacles\n\n`;

        let totalTests = 0;
        let astarWinCount = 0;
        let mlWinCount = 0;
        let tieCount = 0;

        try {
            for (let i = 0; i < obstacleCounts.length; i++) {
                const obstacleCount = obstacleCounts[i];
                
                progressDiv.innerHTML += `\n📦 Testing with ${obstacleCount} obstacles (${i + 1}/${obstacleCounts.length})...\n`;
                progressDiv.scrollTop = progressDiv.scrollHeight;

                totalTests++;
                
                const gridConfig = await this.generateGridWithObstacles(obstacleCount);
                if (!gridConfig) {
                    progressDiv.innerHTML += `  ⚠️ Failed to generate valid grid\n`;
                    continue;
                }

                const algo1Result = await this.runTestWithFixedPoints(algo1, TEST_TIMEOUT, gridConfig);
                const algo2Result = await this.runTestWithFixedPoints(algo2, TEST_TIMEOUT, gridConfig);

                const comparison = {
                    obstacleCount,
                    environmentMode: this.obstacleEnvironmentMode,
                    gridConfig,
                    algo1: {
                        name: algo1,
                        success: algo1Result.success,
                        nodes: algo1Result.nodes,
                        pathLength: algo1Result.distance,
                        path: algo1Result.path || [],
                        time: algo1Result.time
                    },
                    algo2: {
                        name: algo2,
                        success: algo2Result.success,
                        nodes: algo2Result.nodes,
                        pathLength: algo2Result.distance,
                        path: algo2Result.path || [],
                        time: algo2Result.time
                    },
                    winner: 'tie'
                };

                if (algo1Result.success && algo2Result.success) {
                    const algo1Score = algo1Result.nodes + (algo1Result.distance * 10);
                    const algo2Score = algo2Result.nodes + (algo2Result.distance * 10);
                    
                    if (algo2Result.nodes < algo1Result.nodes && algo2Result.distance <= algo1Result.distance) {
                        comparison.winner = 'algo2';
                        mlWinCount++;
                        mlWins.push(comparison);
                        progressDiv.innerHTML += `  ✅ 🎉 ${algo2.toUpperCase()} WINS! (Nodes: ${algo2}=${algo2Result.nodes} vs ${algo1}=${algo1Result.nodes}, Path: ${algo2}=${algo2Result.distance.toFixed(2)} vs ${algo1}=${algo1Result.distance.toFixed(2)})\n`;
                    } else if (algo1Result.nodes < algo2Result.nodes || algo1Result.distance < algo2Result.distance) {
                        comparison.winner = 'algo1';
                        astarWinCount++;
                        progressDiv.innerHTML += `  ⭐ ${algo1.toUpperCase()} wins (Nodes: ${algo1}=${algo1Result.nodes} vs ${algo2}=${algo2Result.nodes}, Path: ${algo1}=${algo1Result.distance.toFixed(2)} vs ${algo2}=${algo2Result.distance.toFixed(2)})\n`;
                    } else {
                        tieCount++;
                        progressDiv.innerHTML += `  🤝 Tie (Nodes: ${algo1Result.nodes}, Path: ${algo1Result.distance.toFixed(2)})\n`;
                    }
                } else if (algo2Result.success && !algo1Result.success) {
                    comparison.winner = 'algo2';
                    mlWinCount++;
                    mlWins.push(comparison);
                    progressDiv.innerHTML += `  ✅ 🎉 ${algo2.toUpperCase()} WINS! (${algo2} found path, ${algo1} failed)\n`;
                } else if (algo1Result.success && !algo2Result.success) {
                    comparison.winner = 'algo1';
                    astarWinCount++;
                    progressDiv.innerHTML += `  ⭐ ${algo1.toUpperCase()} wins (${algo1} found path, ${algo2} failed)\n`;
                } else {
                    progressDiv.innerHTML += `  ❌ Both failed to find path\n`;
                }

                allResults.push(comparison);
                progressDiv.scrollTop = progressDiv.scrollHeight;

                await new Promise(resolve => setTimeout(resolve, 10));
            }

            if (mlWins.length > 0) {
                let tableHtml = `
                    <table>
                        <thead>
                            <tr>
                                <th>Obstacles</th>
                                <th>${algo1.toUpperCase()} Nodes</th>
                                <th>${algo2.toUpperCase()} Nodes</th>
                                <th>${algo1.toUpperCase()} Path</th>
                                <th>${algo2.toUpperCase()} Path</th>
                                <th>Nodes Saved</th>
                                <th>Grid Setup</th>
                                <th>Path Comparison</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                mlWins.forEach((win, index) => {
                    const nodesSaved = win.algo1.nodes - win.algo2.nodes;
                    const nodesPct = ((nodesSaved / win.algo1.nodes) * 100).toFixed(1);
                    const canvasId = `grid-canvas-${index}`;
                    const pathCanvasId = `path-canvas-${index}`;
                    tableHtml += `
                        <tr>
                            <td>${win.obstacleCount}</td>
                            <td>${win.algo1.nodes}</td>
                            <td><strong>${win.algo2.nodes}</strong></td>
                            <td>${win.algo1.pathLength.toFixed(2)}</td>
                            <td><strong>${win.algo2.pathLength.toFixed(2)}</strong></td>
                            <td style="color: #11998e; font-weight: bold;">-${nodesSaved} (${nodesPct}%)</td>
                            <td><canvas id="${canvasId}" width="150" height="150" style="border: 1px solid #ddd; cursor: pointer;" title="Click to enlarge"></canvas></td>
                            <td><canvas id="${pathCanvasId}" width="150" height="150" style="border: 1px solid #ddd; cursor: pointer;" title="Blue: ${algo1.toUpperCase()} Path, Orange: ${algo2.toUpperCase()} Path"></canvas></td>
                        </tr>
                    `;
                });
                tableHtml += '</tbody></table>';
                mlWinsTable.innerHTML = tableHtml;
                
                setTimeout(() => {
                    mlWins.forEach((win, index) => {
                        const canvas = document.getElementById(`grid-canvas-${index}`);
                        const pathCanvas = document.getElementById(`path-canvas-${index}`);
                        if (canvas) {
                            this.drawGridSnapshot(canvas, win.gridConfig);
                            canvas.addEventListener('click', () => {
                                this.showEnlargedGrid(win.gridConfig, win);
                            });
                        }
                        if (pathCanvas) {
                            this.drawPathComparison(pathCanvas, win.gridConfig, win.algo1.path, win.algo2.path);
                            pathCanvas.addEventListener('click', () => {
                                this.showEnlargedPaths(win.gridConfig, win);
                            });
                        }
                    });
                }, 100);
            } else {
                mlWinsTable.innerHTML = `<p style="text-align: center; opacity: 0.8;">No cases found where ${algo2} beat ${algo1} in this test run.</p>`;
            }

            const avgAlgo1Nodes = allResults.filter(r => r.algo1.success).reduce((sum, r) => sum + r.algo1.nodes, 0) / allResults.filter(r => r.algo1.success).length || 0;
            const avgAlgo2Nodes = allResults.filter(r => r.algo2.success).reduce((sum, r) => sum + r.algo2.nodes, 0) / allResults.filter(r => r.algo2.success).length || 0;

            summaryDiv.innerHTML = `
                <h4>📊 Final Summary (${envLabel})</h4>
                <div class="summary-stats-grid">
                    <div class="summary-stat-card">
                        <div class="stat-number">${totalTests}</div>
                        <div class="stat-label">Total Tests</div>
                    </div>
                    <div class="summary-stat-card">
                        <div class="stat-number">${astarWinCount}</div>
                        <div class="stat-label">${algo1.toUpperCase()} Wins</div>
                    </div>
                    <div class="summary-stat-card">
                        <div class="stat-number" style="color: #38ef7d;">${mlWinCount}</div>
                        <div class="stat-label">${algo2.toUpperCase()} Wins 🎉</div>
                    </div>
                    <div class="summary-stat-card">
                        <div class="stat-number">${tieCount}</div>
                        <div class="stat-label">Ties</div>
                    </div>
                    <div class="summary-stat-card">
                        <div class="stat-number">${avgAlgo1Nodes.toFixed(0)}</div>
                        <div class="stat-label">Avg ${algo1.toUpperCase()} Nodes</div>
                    </div>
                    <div class="summary-stat-card">
                        <div class="stat-number">${avgAlgo2Nodes.toFixed(0)}</div>
                        <div class="stat-label">Avg ${algo2.toUpperCase()} Nodes</div>
                    </div>
                </div>
                <div style="margin-top: 24px; padding: 16px; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid #667eea;">
                    <h5 style="margin: 0 0 12px 0; color: #667eea;">📈 Performance Breakdown</h5>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #3b82f6;">${astarWinCount}</div>
                            <div style="font-size: 12px; opacity: 0.8;">A* Better than ML</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #8b5cf6;">${tieCount}</div>
                            <div style="font-size: 12px; opacity: 0.8;">A* Same as ML</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #38ef7d;">${mlWinCount}</div>
                            <div style="font-size: 12px; opacity: 0.8;">ML Better than A*</div>
                        </div>
                    </div>
                </div>
                <p style="text-align: center; margin-top: 16px; font-size: 18px;">
                    ${mlWinCount > astarWinCount ? '🏆 ML performed better overall!' : 
                      astarWinCount > mlWinCount ? '⭐ A* performed better overall.' : 
                      '🤝 Both algorithms performed equally.'}
                </p>
            `;

            progressDiv.innerHTML += `\n✅ Test complete! Ran ${totalTests} comparisons.\n`;
            progressDiv.innerHTML += `   A* wins: ${astarWinCount}, ML wins: ${mlWinCount}, Ties: ${tieCount}\n`;

            this.obstacleResults = allResults;
            
            this.obstacleSummaryStats = {
                totalTests: totalTests,
                astarWins: astarWinCount,
                mlWins: mlWinCount,
                ties: tieCount,
                avgAstarNodes: avgAlgo1Nodes,
                avgMlNodes: avgAlgo2Nodes
            };
            
            this.obstacleMlWins = mlWins;
            
            const exportObstacleBtn = document.getElementById('exportObstacleResultsBtn');
            if (exportObstacleBtn) {
                exportObstacleBtn.disabled = false;
            }

        } catch (error) {
            console.error('Obstacle comparison error:', error);
            progressDiv.innerHTML += `\n❌ Error: ${error.message}\n`;
        } finally {
            this.running = false;
            if (obstacleBtn) obstacleBtn.disabled = false;
        }
    }

    generateGridWithObstacles = async (obstacleCount) => {
        const rows = this.gridGenerator.grid.rows;
        const cols = this.gridGenerator.grid.cols;
        const totalCells = rows * cols;

        // Clear grid first
        this.gridGenerator.clearGrid();

        // Get all cell positions
        const allPositions = [];
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                allPositions.push({ x, y });
            }
        }

        // Shuffle positions
        for (let i = allPositions.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [allPositions[i], allPositions[j]] = [allPositions[j], allPositions[i]];
        }

        // Select random start and end (first two non-obstacle positions)
        const startPos = allPositions[0];
        const endPos = allPositions[1];

        // Ensure start and end are far enough apart
        const minDist = Math.floor(Math.sqrt(rows * rows + cols * cols) * 0.3);
        let actualEnd = endPos;
        for (let i = 2; i < allPositions.length; i++) {
            const dist = Math.sqrt(
                Math.pow(allPositions[i].x - startPos.x, 2) + 
                Math.pow(allPositions[i].y - startPos.y, 2)
            );
            if (dist >= minDist) {
                actualEnd = allPositions[i];
                break;
            }
        }

        // Place obstacles (skip start and end positions)
        const walls = [];
        let placed = 0;
        for (let i = 2; i < allPositions.length && placed < obstacleCount; i++) {
            const pos = allPositions[i];
            if ((pos.x === startPos.x && pos.y === startPos.y) || 
                (pos.x === actualEnd.x && pos.y === actualEnd.y)) {
                continue;
            }
            this.gridGenerator.grid.grid[pos.y][pos.x].isWall = true;
            walls.push(pos);
            placed++;
        }

        // Get valid positions (non-walls)
        const validPositions = [];
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                if (!this.gridGenerator.grid.grid[y][x].isWall) {
                    validPositions.push(this.gridGenerator.grid.grid[y][x]);
                }
            }
        }

        // Find start and end cells
        const start = this.gridGenerator.grid.grid[startPos.y][startPos.x];
        const end = this.gridGenerator.grid.grid[actualEnd.y][actualEnd.x];

        return {
            walls,
            valid: validPositions,
            start,
            end
        };
    }

    runTestWithFixedPoints = async (algorithm, timeoutMs, gridConfig) => {
        const startTime = performance.now();
        
        try {
            // Clear the grid
            this.gridGenerator.clearGrid();
            
            // Restore walls from configuration
            for (const wall of gridConfig.walls) {
                this.gridGenerator.grid.grid[wall.y][wall.x].isWall = true;
            }

            // Use the fixed start/end from gridConfig
            const start = gridConfig.start;
            const end = gridConfig.end;

            if (!start || !end) {
                throw new Error('gridConfig must have start and end points');
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
                path: result.success ? result.path : [],  // Return the actual path
                time: performance.now() - startTime
            };
        } catch (error) {
            console.error('Error in runTestWithFixedPoints:', error);
            return {
                success: false,
                nodes: 0,
                distance: 0,
                path: [],
                time: performance.now() - startTime
            };
        }
    }

    runSingleTestWithLayoutAndPoints = async (algorithm, timeoutMs, gridConfig) => {
        const startTime = performance.now();
        
        try {
            // Clear the grid
            this.gridGenerator.clearGrid();
            
            // Restore walls from configuration
            for (const wall of gridConfig.walls) {
                this.gridGenerator.grid.grid[wall.y][wall.x].isWall = true;
            }

            const start = gridConfig.start;
            const end = gridConfig.end;

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

    drawGridSnapshot = (canvas, gridConfig) => {
        const ctx = canvas.getContext('2d');
        const rows = this.gridGenerator.grid.rows;
        const cols = this.gridGenerator.grid.cols;
        const cellWidth = canvas.width / cols;
        const cellHeight = canvas.height / rows;

        // Clear canvas
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw walls
        ctx.fillStyle = '#333333';
        gridConfig.walls.forEach(wall => {
            ctx.fillRect(wall.x * cellWidth, wall.y * cellHeight, cellWidth, cellHeight);
        });

        // Draw start point (green)
        if (gridConfig.start) {
            ctx.fillStyle = '#4ade80';
            ctx.beginPath();
            ctx.arc(
                (gridConfig.start.col + 0.5) * cellWidth,
                (gridConfig.start.row + 0.5) * cellHeight,
                Math.min(cellWidth, cellHeight) * 0.3,
                0, 2 * Math.PI
            );
            ctx.fill();
        }

        // Draw end point (red)
        if (gridConfig.end) {
            ctx.fillStyle = '#f87171';
            ctx.beginPath();
            ctx.arc(
                (gridConfig.end.col + 0.5) * cellWidth,
                (gridConfig.end.row + 0.5) * cellHeight,
                Math.min(cellWidth, cellHeight) * 0.3,
                0, 2 * Math.PI
            );
            ctx.fill();
        }

        // Draw grid lines
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= cols; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellWidth, 0);
            ctx.lineTo(i * cellWidth, canvas.height);
            ctx.stroke();
        }
        for (let i = 0; i <= rows; i++) {
            ctx.beginPath();
            ctx.moveTo(0, i * cellHeight);
            ctx.lineTo(canvas.width, i * cellHeight);
            ctx.stroke();
        }
    }

    showEnlargedGrid = (gridConfig, winData) => {
        // Create modal overlay
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            cursor: pointer;
        `;

        // Create content container
        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            border-radius: 12px;
            padding: 20px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            cursor: default;
        `;
        content.onclick = (e) => e.stopPropagation();

        // Add title
        const title = document.createElement('h3');
        title.textContent = `ML Win - ${winData.obstacleCount} Obstacles (Run ${winData.run})`;
        title.style.marginTop = '0';
        content.appendChild(title);

        // Add stats
        const stats = document.createElement('div');
        stats.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;';
        stats.innerHTML = `
            <div style="padding: 8px; background: #f0f4f8; border-radius: 6px;">
                <strong>A*:</strong> ${winData.astar.nodes} nodes, path: ${winData.astar.pathLength.toFixed(2)}
            </div>
            <div style="padding: 8px; background: #d1fae5; border-radius: 6px;">
                <strong>ML:</strong> ${winData.ml.nodes} nodes, path: ${winData.ml.pathLength.toFixed(2)}
            </div>
        `;
        content.appendChild(stats);

        // Add large canvas
        const canvas = document.createElement('canvas');
        canvas.width = 500;
        canvas.height = 500;
        canvas.style.border = '2px solid #ddd';
        content.appendChild(canvas);

        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Close';
        closeBtn.style.cssText = `
            margin-top: 16px;
            padding: 8px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        `;
        closeBtn.onclick = () => document.body.removeChild(modal);
        content.appendChild(closeBtn);

        modal.appendChild(content);
        modal.onclick = () => document.body.removeChild(modal);
        document.body.appendChild(modal);

        // Draw the grid
        this.drawGridSnapshot(canvas, gridConfig);
    }

    drawPathComparison = (canvas, gridConfig, astarPath, mlPath) => {
        const ctx = canvas.getContext('2d');
        const rows = this.gridGenerator.grid.rows;
        const cols = this.gridGenerator.grid.cols;
        const cellWidth = canvas.width / cols;
        const cellHeight = canvas.height / rows;

        // Clear canvas with white background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw walls
        ctx.fillStyle = '#333333';
        gridConfig.walls.forEach(wall => {
            ctx.fillRect(wall.x * cellWidth, wall.y * cellHeight, cellWidth, cellHeight);
        });

        // Draw A* path (blue)
        if (astarPath && astarPath.length > 0) {
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            astarPath.forEach((cell, idx) => {
                const x = (cell.col + 0.5) * cellWidth;
                const y = (cell.row + 0.5) * cellHeight;
                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();

            // Draw A* path points
            ctx.fillStyle = '#3b82f6';
            astarPath.forEach(cell => {
                ctx.beginPath();
                ctx.arc(
                    (cell.col + 0.5) * cellWidth,
                    (cell.row + 0.5) * cellHeight,
                    2,
                    0, 2 * Math.PI
                );
                ctx.fill();
            });
        }

        // Draw ML path (orange)
        if (mlPath && mlPath.length > 0) {
            ctx.strokeStyle = '#f97316';
            ctx.lineWidth = 2;
            ctx.beginPath();
            mlPath.forEach((cell, idx) => {
                const x = (cell.col + 0.5) * cellWidth;
                const y = (cell.row + 0.5) * cellHeight;
                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();

            // Draw ML path points
            ctx.fillStyle = '#f97316';
            mlPath.forEach(cell => {
                ctx.beginPath();
                ctx.arc(
                    (cell.col + 0.5) * cellWidth,
                    (cell.row + 0.5) * cellHeight,
                    2,
                    0, 2 * Math.PI
                );
                ctx.fill();
            });
        }

        // Draw start point (green) - larger to be visible
        if (gridConfig.start) {
            ctx.fillStyle = '#4ade80';
            ctx.beginPath();
            ctx.arc(
                (gridConfig.start.col + 0.5) * cellWidth,
                (gridConfig.start.row + 0.5) * cellHeight,
                Math.min(cellWidth, cellHeight) * 0.35,
                0, 2 * Math.PI
            );
            ctx.fill();
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Draw end point (red) - larger to be visible
        if (gridConfig.end) {
            ctx.fillStyle = '#f87171';
            ctx.beginPath();
            ctx.arc(
                (gridConfig.end.col + 0.5) * cellWidth,
                (gridConfig.end.row + 0.5) * cellHeight,
                Math.min(cellWidth, cellHeight) * 0.35,
                0, 2 * Math.PI
            );
            ctx.fill();
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Draw grid lines
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= cols; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellWidth, 0);
            ctx.lineTo(i * cellWidth, canvas.height);
            ctx.stroke();
        }
        for (let i = 0; i <= rows; i++) {
            ctx.beginPath();
            ctx.moveTo(0, i * cellHeight);
            ctx.lineTo(canvas.width, i * cellHeight);
            ctx.stroke();
        }
    }

    showEnlargedPaths = (gridConfig, winData) => {
        // Create modal overlay
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            cursor: pointer;
        `;

        // Create content container
        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            border-radius: 12px;
            padding: 20px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            cursor: default;
        `;
        content.onclick = (e) => e.stopPropagation();

        // Add title
        const title = document.createElement('h3');
        title.textContent = `Path Comparison - ${winData.obstacleCount} Obstacles (Run ${winData.run})`;
        title.style.marginTop = '0';
        content.appendChild(title);

        // Add legend
        const legend = document.createElement('div');
        legend.style.cssText = 'display: flex; gap: 24px; justify-content: center; margin-bottom: 16px;';
        legend.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 3px; background: #3b82f6;"></div>
                <span>A* Path (${winData.astar.nodes} nodes)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 3px; background: #f97316;"></div>
                <span>ML Path (${winData.ml.nodes} nodes)</span>
            </div>
        `;
        content.appendChild(legend);

        // Add stats
        const stats = document.createElement('div');
        stats.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;';
        const nodesSaved = winData.astar.nodes - winData.ml.nodes;
        const pathDiff = (winData.ml.pathLength - winData.astar.pathLength).toFixed(2);
        stats.innerHTML = `
            <div style="padding: 8px; background: #dbeafe; border-radius: 6px; border: 2px solid #3b82f6;">
                <strong>A*:</strong> ${winData.astar.nodes} nodes, path: ${winData.astar.pathLength.toFixed(2)}
            </div>
            <div style="padding: 8px; background: #fed7aa; border-radius: 6px; border: 2px solid #f97316;">
                <strong>ML:</strong> ${winData.ml.nodes} nodes, path: ${winData.ml.pathLength.toFixed(2)}
            </div>
        `;
        content.appendChild(stats);

        // Add improvement summary
        const improvement = document.createElement('div');
        improvement.style.cssText = 'text-align: center; padding: 12px; background: #d1fae5; border-radius: 6px; margin-bottom: 16px; font-weight: bold; color: #065f46;';
        improvement.textContent = `ML saved ${nodesSaved} nodes (${((nodesSaved / winData.astar.nodes) * 100).toFixed(1)}% reduction)`;
        content.appendChild(improvement);

        // Add large canvas
        const canvas = document.createElement('canvas');
        canvas.width = 600;
        canvas.height = 600;
        canvas.style.border = '2px solid #ddd';
        content.appendChild(canvas);

        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Close';
        closeBtn.style.cssText = `
            margin-top: 16px;
            padding: 8px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        `;
        closeBtn.onclick = () => document.body.removeChild(modal);
        content.appendChild(closeBtn);

        modal.appendChild(content);
        modal.onclick = () => document.body.removeChild(modal);
        document.body.appendChild(modal);

        // Draw the paths
        this.drawPathComparison(canvas, gridConfig, winData.astar.path, winData.ml.path);
    }
}
