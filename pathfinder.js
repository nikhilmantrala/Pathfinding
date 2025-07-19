export class Pathfinder {
    constructor(grid, heuristicFn) {
        this.grid = grid;
        this.heuristic = heuristicFn;
        this.nodesVisited = 0;
        this.distanceTraveled = 0;
    }
    async run(start, end, onStep, onFinish) {
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                const cell = this.grid.grid[i][j];
                cell.g = Infinity;
                cell.f = Infinity;
                cell.h = 0;
                cell.parent = null;
            }
        }
        

        start.g = 0;
        start.f = await this.heuristic(start, end);
        start.color = "green";
        end.color = "red";
        const openSet = [start];
        this.nodesVisited = 0;
        this.distanceTraveled = 0;


        const getNeighbors = (cell) => {
            let neighbors = [];
            let directions = [ //possible directions of travel including diagonals
                { row: -1, col: 0 }, { row: 1, col: 0 }, { row: 0, col: -1 }, { row: 0, col: 1 },
                { row: -1, col: -1 }, { row: -1, col: 1 }, { row: 1, col: -1 }, { row: 1, col: 1 }
            ];
            for (let dir of directions) {
                let dx = dir.row, dy = dir.col;
                const r = cell.row + dx, c = cell.col + dy;
                if (r >= 0 && r < this.grid.rows && c >= 0 && c < this.grid.cols) {
                    const neighbor = this.grid.grid[r][c];
                    if (neighbor.isWall) continue;
                    if (Math.abs(dx) === 1 && Math.abs(dy) === 1) {
                        if (this.grid.grid[cell.row][cell.col + dy]?.isWall || this.grid.grid[cell.row + dx][cell.col]?.isWall) continue;
                    }
                    neighbors.push(neighbor);
                }
            }
            return neighbors;


        };
        const calculateCosts = (current, neighbor, end) => { //Calculating the costs of moving to a neighbor
            const dx = Math.abs(neighbor.row - current.row);
            const dy = Math.abs(neighbor.col - current.col);
            const stepCost = (dx === 1 && dy === 1) ? Math.SQRT2 : 1; // diagonal moves cost more
            const tempG = current.g + stepCost * neighbor.weight;
            const h = this.heuristic(neighbor, end); //heuristic cost
            const f = tempG + h; //total cost
            return { tempG, h, f };
        };
        const step = () => {
            if (openSet.length === 0) {
                onFinish(false, this.nodesVisited, 'No path');
                return;
            }
            openSet.sort((a, b) => a.f - b.f);
            const current = openSet.shift();
            this.nodesVisited++;
            if (current === end) {
                this.distanceTraveled = current.g;
                onFinish(true, this.nodesVisited, this.distanceTraveled.toFixed(2));
                return;
            }
            for (const neighbor of getNeighbors(current)) {
                const { tempG, h, f } = calculateCosts(current, neighbor, end);
                if (tempG < neighbor.g) {
                    neighbor.parent = current;
                    neighbor.g = tempG;
                    neighbor.h = h;
                    neighbor.f = f;
                    if (!openSet.includes(neighbor)) {
                        openSet.push(neighbor);
                        neighbor.color = "yellow";
                    }
                }
            }
            if (current !== start) current.color = "blue";
            start.color = "green";
            end.color = "red";
            if (onStep) onStep();
            requestAnimationFrame(step);
        };
        step();
   }
    // Async version for ML heuristic
    async runAsync(start, end, onStep, onFinish) {
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                const cell = this.grid.grid[i][j];
                cell.g = Infinity;
                cell.f = Infinity;
                cell.h = 0;
                cell.parent = null;
            }
        }
        start.g = 0;
        start.f = await this.heuristic(start, end);
        start.color = "green";
        end.color = "red";
        const openSet = [start];
        this.nodesVisited = 0;
        this.distanceTraveled = 0;
        const getNeighbors = (cell) => {
            let neighbors = [];
            let directions = [
                { row: -1, col: 0 }, { row: 1, col: 0 }, { row: 0, col: -1 }, { row: 0, col: 1 },
                { row: -1, col: -1 }, { row: -1, col: 1 }, { row: 1, col: -1 }, { row: 1, col: 1 }
            ];
            for (let dir of directions) {
                let dx = dir.row, dy = dir.col;
                const r = cell.row + dx, c = cell.col + dy;
                if (r >= 0 && r < this.grid.rows && c >= 0 && c < this.grid.cols) {
                    const neighbor = this.grid.grid[r][c];
                    if (neighbor.isWall) continue;
                    if (Math.abs(dx) === 1 && Math.abs(dy) === 1) {
                        if (this.grid.grid[cell.row][cell.col + dy]?.isWall || this.grid.grid[cell.row + dx][cell.col]?.isWall) continue;
                    }
                    neighbors.push(neighbor);
                }
            }
            return neighbors;
        };
        const calculateCosts = async (current, neighbor, end) => {
            const dx = Math.abs(neighbor.row - current.row);
            const dy = Math.abs(neighbor.col - current.col);
            const stepCost = (dx === 1 && dy === 1) ? Math.SQRT2 : 1;
            const tempG = current.g + stepCost * neighbor.weight;
            const h = await this.heuristic(neighbor, end);
            const f = tempG + h;
            return { tempG, h, f };
        };
        const step = async () => {
            if (openSet.length === 0) {
                onFinish(false, this.nodesVisited, 'No path');
                return;
            }
            openSet.sort((a, b) => a.f - b.f);
            const current = openSet.shift();
            this.nodesVisited++;
            if (current === end) {
                this.distanceTraveled = current.g;
                onFinish(true, this.nodesVisited, this.distanceTraveled.toFixed(2));
                return;
            }
            for (const neighbor of getNeighbors(current)) {
                const { tempG, h, f } = await calculateCosts(current, neighbor, end);
                if (tempG < neighbor.g) {
                    neighbor.parent = current;
                    neighbor.g = tempG;
                    neighbor.h = h;
                    neighbor.f = f;
                    if (!openSet.includes(neighbor)) {
                        openSet.push(neighbor);
                        neighbor.color = "yellow";
                    }
                }
            }
            if (current !== start) current.color = "blue";
            start.color = "green";
            end.color = "red";
            if (onStep) onStep();
            setTimeout(step, 0); // async step
        };
        await step();
    }
}



let mlModel = null;
export async function loadMLModel() {
    if (!mlModel) {
        // Ensure TensorFlow.js is ready with CPU backend
        await window.tf.ready();
        
        mlModel = await window.tf.loadGraphModel('web_model_static/model.json?v=' + Date.now());
        if (mlModel && mlModel.weights && mlModel.weights.length) {
            console.log("Loaded static model weights count:", mlModel.weights.length);
        }
        if (mlModel && mlModel.inputs) {
            console.log("Static Model input names:", mlModel.inputs.map(x => x.name));
        }
    }
    return mlModel;
}

export async function mlHeuristic(a, b, grid) {
    await loadMLModel();
    const gridArr = [];
    for (let i = 0; i < grid.length; i++) {
        const row = [];
        for (let j = 0; j < grid[0].length; j++) {
            row.push(grid[i][j].isWall ? 1 : 0);
        }
        gridArr.push(row);
    }
    const gridTensor = window.tf.tensor(gridArr).reshape([1, grid.length, grid[0].length, 1]);
    const startGoalArr = [
        a.row / (grid.length - 1),
        a.col / (grid[0].length - 1),
        b.row / (grid.length - 1),
        b.col / (grid[0].length - 1)
    ];
    const startGoalTensor = window.tf.tensor(startGoalArr).reshape([1, 4]);

    console.log('gridTensor shape:', gridTensor.shape, 'startGoalTensor shape:', startGoalTensor.shape);
    let output;
    let errorGuard = false;
    if (mlModel.predict) {
        // Model expects inputs in order: [start_goal, grid] based on signature
        output = mlModel.predict([startGoalTensor, gridTensor]);
    } else {
        const inputNames = mlModel.inputs.map(x => x.name);
        
        console.log('DEBUG: Model input names:', inputNames);
        console.log('DEBUG: gridTensor shape:', gridTensor.shape);
        console.log('DEBUG: startGoalTensor shape:', startGoalTensor.shape);
        
        // Try using the internal names from the model inputs array
        const inputDict = {};
        inputNames.forEach(name => {
            if (name === 'start_goal' || name === 'start_goal:0') {
                inputDict[name] = startGoalTensor;
                console.log(`DEBUG: Mapped ${name} to startGoalTensor with shape:`, startGoalTensor.shape);
            } else if (name === 'grid' || name === 'grid:0') {
                inputDict[name] = gridTensor;
                console.log(`DEBUG: Mapped ${name} to gridTensor with shape:`, gridTensor.shape);
            }
        });
        
        console.log('DEBUG: Final inputDict keys:', Object.keys(inputDict));
        console.log('DEBUG: Final inputDict shapes:', Object.fromEntries(Object.entries(inputDict).map(([k,v]) => [k, v.shape])));
        
        // Double-check the tensor assignment
        if (inputDict['start_goal'] && inputDict['start_goal'].shape.toString() !== '1,4') {
            console.error('ERROR: start_goal tensor has wrong shape!', inputDict['start_goal'].shape);
        }
        if (inputDict['grid'] && inputDict['grid'].shape.toString() !== '1,20,20,1') {
            console.error('ERROR: grid tensor has wrong shape!', inputDict['grid'].shape);
        }
        
        // Check for missing inputs
        if (Object.keys(inputDict).length !== inputNames.length) {
            if (!errorGuard) {
                errorGuard = true;
                console.error('Static ML: Could not map all inputs. Expected names:', inputNames);
            }
            gridTensor.dispose();
            startGoalTensor.dispose();
            throw new Error('Static ML: Could not map all model inputs.');
        }
        console.log('Static ML inputNames:', inputNames);
        console.log('Static ML inputDict mapping:', Object.fromEntries(Object.entries(inputDict).map(([k,v]) => [k, v.shape])));
        
        console.log('DEBUG: About to call mlModel.execute with inputDict');
        output = mlModel.execute(inputDict);
    }
    const normResidual = (await output.data())[0];
    gridTensor.dispose();
    startGoalTensor.dispose();
    output.dispose();
   
    const maxCost = Math.SQRT2 * (grid.length - 1);
    const dx = Math.abs(a.row - b.row);
    const dy = Math.abs(a.col - b.col);
    const octile = Math.min(dx, dy) * Math.SQRT2 + Math.abs(dx - dy);
    const predictedCost = normResidual * maxCost + octile;
    const admissibleCost = Math.min(predictedCost, octile);
    console.log("ML residual (norm):", normResidual, "predicted cost:", predictedCost, "octile:", octile, "admissible:", admissibleCost);
    return admissibleCost;
}

export function AstarHeuristic(a, b) {
    const dx = Math.abs(a.row - b.row);
    const dy = Math.abs(a.col - b.col);
    const D = 1;
    const D2 = Math.SQRT2;
    
    return D * (dx + dy) + (D2 - 2 * D) * Math.min(dx, dy);
}

export function dijkastraHeuristic(a,b) {
    return 0;
}

export function greedyHeuristic(a, b) {
    return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
}

export async function loadMLDynamicModel() {
    if (!window.mlDynamicModel) {
        // Ensure TensorFlow.js is ready with CPU backend
        await window.tf.ready();
        
        window.mlDynamicModel = await window.tf.loadGraphModel('web_model_dynamic/model.json?v=' + Date.now());
        if (window.mlDynamicModel && window.mlDynamicModel.inputs) {
            console.log("Dynamic Model input names:", window.mlDynamicModel.inputs.map(x => x.name));
        }
    }
    return window.mlDynamicModel;
}

export async function mlDynamicHeuristic(a, b, grid) {
    await loadMLDynamicModel();
    const gridArr = [];
    const costArr = [];
    for (let i = 0; i < grid.length; i++) {
        const gridRow = [];
        const costRow = [];
        for (let j = 0; j < grid[0].length; j++) {
            gridRow.push(grid[i][j].isWall ? 1 : 0);
            costRow.push(grid[i][j].cost ?? 1);
        }
        gridArr.push(gridRow);
        costArr.push(costRow);
    }
    const gridTensor = window.tf.tensor(gridArr).reshape([1, grid.length, grid[0].length, 1]);
    const costTensor = window.tf.tensor(costArr).reshape([1, grid.length, grid[0].length, 1]);
    const startGoalArr = [
        a.row / (grid.length - 1),
        a.col / (grid[0].length - 1),
        b.row / (grid.length - 1),
        b.col / (grid[0].length - 1)
    ];
    const startGoalTensor = window.tf.tensor(startGoalArr).reshape([1, 4]);
    let output;
    let errorGuard = false;
    if (window.mlDynamicModel.predict) {
        // Model expects inputs in order: [start_goal, grid:0, cost:0] based on signature
        output = window.mlDynamicModel.predict([startGoalTensor, gridTensor, costTensor]);
    } else {
        const inputNames = window.mlDynamicModel.inputs.map(x => x.name);
        
        // Use signature input names - the dynamic model has inconsistent naming:
        // "start_goal" (no :0), "grid:0" (with :0), "cost:0" (with :0)
        const inputDict = {
            'start_goal': startGoalTensor,
            'grid:0': gridTensor,
            'cost:0': costTensor
        };
        
        // Check for missing inputs
        const expectedKeys = ['start_goal', 'grid:0', 'cost:0'];
        const missingInputs = expectedKeys.filter(key => !inputDict[key]);
        if (missingInputs.length > 0) {
            if (!errorGuard) {
                errorGuard = true;
                console.error('Dynamic ML: Missing inputs:', missingInputs, 'Actual input names from model:', inputNames);
            }
            gridTensor.dispose();
            costTensor.dispose();
            startGoalTensor.dispose();
            throw new Error('Dynamic ML: Missing required inputs: ' + missingInputs.join(', '));
        }
        console.log('Dynamic ML inputNames:', inputNames);
        console.log('Dynamic ML inputDict mapping:', Object.fromEntries(Object.entries(inputDict).map(([k,v]) => [k, v.shape])));
        output = window.mlDynamicModel.execute(inputDict);
    }
    const normResidual = (await output.data())[0];
    gridTensor.dispose();
    costTensor.dispose();
    startGoalTensor.dispose();
    output.dispose();
    const maxCost = Math.SQRT2 * (grid.length - 1) * 3.0;
    const dx = Math.abs(a.row - b.row);
    const dy = Math.abs(a.col - b.col);
    const octile = Math.min(dx, dy) * Math.SQRT2 + Math.abs(dx - dy);
    const predictedCost = normResidual * maxCost + octile;
    const admissibleCost = Math.min(predictedCost, octile);
    console.log("ML-Dynamic residual (norm):", normResidual, "predicted cost:", predictedCost, "octile:", octile, "admissible:", admissibleCost);
    return admissibleCost;
}

