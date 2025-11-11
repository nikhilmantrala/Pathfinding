export class UIManager {
    constructor(canvas, ROWS, COLS) {
        this.canvas = canvas;
        this.ROWS = ROWS;
        this.COLS = COLS;
        this.WIDTH = canvas.width;
        this.HEIGHT = canvas.height;
        this.cellSize = this.WIDTH / ROWS;
        this.initializeElements();
    }

    initializeElements() {
        this.elements = {
            algoSelect: document.getElementById('algoSelect'),
            gridTypeSelect: document.getElementById('gridTypeSelect'),
            results: document.getElementById('resultsBody'),
            setupName: document.getElementById('setupName'),
            setupSelect: document.getElementById('setupSelect'),
            batchTest: {
                algorithmPairSelect: document.getElementById('algorithmPairSelect'),
                layoutFamilies: document.getElementById('layoutFamilies'),
                densityBands: document.getElementById('densityBands'),
                seedPreset: document.getElementById('seedPreset'),
                pairsPerSeed: document.getElementById('pairsPerSeed'),
                timeoutMs: document.getElementById('timeoutMs'),
                saveInputs: document.getElementById('saveInputs'),
                progress: document.getElementById('batchTestProgress'),
                progressBar: document.querySelector('#batchTestProgress .progress-fill'),
                progressText: document.querySelector('#batchTestProgress .progress-text'),
                exportBtn: document.getElementById('exportResultsBtn')
            },
            controls: {
                startBtn: document.getElementById('startBtn'),
                clearBtn: document.getElementById('clearBtn'),
                clearDynamicBtn: document.getElementById('clearDynamicBtn'),
                generateBtn: document.getElementById('generateBtn'),
                batchTestBtn: document.getElementById('batchTestBtn'),
                saveBtn: document.getElementById('saveBtn'),
                loadBtn: document.getElementById('loadBtn'),
                exportBtn: document.getElementById('exportBtn')
            }
        };
    }

    drawGrid(grid, showCosts = false) {
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, this.WIDTH, this.HEIGHT);

        for (let i = 0; i < this.ROWS; i++) {
            for (let j = 0; j < this.COLS; j++) {
                const cell = grid[i][j];
                ctx.fillStyle = cell.color;
                ctx.fillRect(j * this.cellSize, i * this.cellSize, this.cellSize, this.cellSize);
                ctx.strokeStyle = '#aaa';
                ctx.lineWidth = 1;
                ctx.strokeRect(j * this.cellSize, i * this.cellSize, this.cellSize, this.cellSize);
                
                if (showCosts && !cell.isWall) {
                    ctx.fillStyle = 'black';
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(
                        cell.cost.toFixed(1), 
                        j * this.cellSize + this.cellSize/2, 
                        i * this.cellSize + this.cellSize/2
                    );
                }
            }
        }
    }

    updateResults(runHistory) {
        if (!this.elements.results) return;
        
        const tbody = this.elements.results;
        tbody.innerHTML = '';
        
        // The table headers in the HTML are expected to be in this order:
        // Test Case | Layout Type | Wall Density | Algorithm | Time (ms) | Nodes Visited | Path Length | Success
        runHistory.forEach((run, i) => {
            const tr = document.createElement('tr');

            const testCase = i + 1;
            const layoutType = run.layoutType || run.layout || 'N/A';
            const wallDensity = (typeof run.wallDensity === 'number') ? run.wallDensity : (run.wallDensity || 'N/A');
            const algo = run.algorithm || run.algo || 'unknown';
            const time = (typeof run.timeMs === 'number') ? run.timeMs.toFixed(2) : (run.timeMs || 'N/A');
            const nodes = (typeof run.nodesVisited === 'number') ? run.nodesVisited : (run.nodesVisited || 'N/A');
            const dist = (typeof run.distanceTraveled === 'number') ? run.distanceTraveled.toFixed(2) : (run.distanceTraveled || 'N/A');
            const success = (typeof run.success === 'boolean') ? (run.success ? '\u2713' : '\u2717') : (run.distanceTraveled ? '\u2713' : '\u2717');

            tr.innerHTML = `
                <td>${testCase}</td>
                <td>${layoutType}</td>
                <td>${wallDensity}</td>
                <td>${algo}</td>
                <td>${time}</td>
                <td>${nodes}</td>
                <td>${dist}</td>
                <td>${success}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    createResultRow() {
        if (!this.elements.results) return null;
        const tr = document.createElement('tr');
        this.elements.results.appendChild(tr);
        return tr;
    }

    disableControls() {
        Object.values(this.elements.controls).forEach(control => {
            if (control) control.disabled = true;
        });
    }

    enableControls() {
        Object.values(this.elements.controls).forEach(control => {
            if (control) control.disabled = false;
        });
    }

    updateSetupDropdown(setups) {
        const select = this.elements.setupSelect;
        if (!select) return;
        
        select.innerHTML = '';
        Object.keys(setups).forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        });
    }

    exportToCSV(data) {
        const csv = data.map(row => row.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'pathfinding_results.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    setupExportButton(batchTestManager) {
        const exportBtn = this.elements.batchTest.exportBtn;
        if (!exportBtn) return;

        // Initially disable the export button
        exportBtn.disabled = true;

        // Add click handler for export button
        exportBtn.addEventListener('click', () => {
            if (batchTestManager?.testResults?.length > 0) {
                batchTestManager.exportToExcel();
            }
        });

        // Return function to update button state
        return () => {
            exportBtn.disabled = !(batchTestManager?.testResults?.length > 0);
        };
    }
}
