export class BatchTestResult {
    constructor() {
        this.layout_family = '';
        this.density_band = '';
        this.seed_layout = 0;
        this.seed_costs = null;
        this.pair_id = 0;
        this.start_r = 0;
        this.start_c = 0;
        this.goal_r = 0;
        this.goal_c = 0;
        this.algorithm = '';
        this.success = 0;
        this.path_cost = null;
        this.nodes_expanded = 0;
        this.runtime_ms = 0;
    }

    static toCsvHeader() {
        return [
            'layout_family',
            'density_band',
            'seed_layout',
            'seed_costs',
            'pair_id',
            'start_r',
            'start_c',
            'goal_r',
            'goal_c',
            'algorithm',
            'success',
            'path_cost',
            'nodes_expanded',
            'runtime_ms'
        ].join(',');
    }

    toCsvRow() {
        return [
            this.layout_family,
            this.density_band,
            this.seed_layout,
            this.seed_costs || '',
            this.pair_id,
            this.start_r,
            this.start_c,
            this.goal_r,
            this.goal_c,
            this.algorithm,
            this.success,
            this.path_cost || '',
            this.nodes_expanded,
            this.runtime_ms.toFixed(2)
        ].join(',');
    }
}

export class BatchTestRunner {
    constructor(config, pathfinder, gridGenerator) {
        this.config = config;
        this.pathfinder = pathfinder;
        this.gridGenerator = gridGenerator;
        this.results = [];
        this.onProgress = null;
        this.onComplete = null;
    }

    getDensityBand(grid) {
        let wallCount = 0;
        for (let row of grid) {
            for (let cell of row) {
                if (cell.isWall) wallCount++;
            }
        }
        const density = wallCount / (grid.length * grid[0].length);
        
        if (density < 0.2) return 'Low';
        if (density < 0.4) return 'Medium';
        return 'High';
    }

    async run() {
        const totalRuns = this.calculateTotalRuns();
        let completedRuns = 0;

        for (const layoutFamily of this.config.selectedLayouts) {
            for (let seedIdx = 0; seedIdx < this.config.seedsPerFamily; seedIdx++) {
                // Generate layout with current seed
                const layoutSeed = Math.floor(Math.random() * 1000000);
                this.gridGenerator.setSeed(layoutSeed);
                this.gridGenerator.generate(layoutFamily.toLowerCase());
                
                const densityBand = this.getDensityBand(this.gridGenerator.grid);
                if (!this.config.selectedDensities.has(densityBand)) continue;

                // Generate cost map for dynamic algorithms
                const costSeed = Math.floor(Math.random() * 1000000);
                
                for (let pairId = 1; pairId <= this.config.pairsPerSeed; pairId++) {
                    const { start, end } = this.generateStartEndPair();
                    
                    for (const algorithm of this.config.selectedPair.algorithms) {
                        const result = await this.runSingleTest(
                            algorithm, 
                            start, 
                            end, 
                            layoutFamily,
                            densityBand,
                            layoutSeed,
                            algorithm.includes('dynamic') ? costSeed : null,
                            pairId
                        );
                        
                        this.results.push(result);
                        completedRuns++;
                        
                        if (this.onProgress) {
                            this.onProgress(completedRuns / totalRuns);
                        }
                    }
                }
            }
        }

        if (this.onComplete) {
            this.onComplete(this.results);
        }
    }

    calculateTotalRuns() {
        const layoutCount = this.config.selectedLayouts.size;
        const algorithmCount = this.config.selectedPair.algorithms.length;
        return layoutCount * this.config.seedsPerFamily * this.config.pairsPerSeed * algorithmCount;
    }

    generateStartEndPair() {
        // Implementation of random start/end point generation
        // Make sure they're valid and not walls
        // Return { start: {r, c}, end: {r, c} }
    }

    async runSingleTest(algorithm, start, end, layoutFamily, densityBand, layoutSeed, costSeed, pairId) {
        const result = new BatchTestResult();
        result.layout_family = layoutFamily;
        result.density_band = densityBand;
        result.seed_layout = layoutSeed;
        result.seed_costs = costSeed;
        result.pair_id = pairId;
        result.start_r = start.r;
        result.start_c = start.c;
        result.goal_r = end.r;
        result.goal_c = end.c;
        result.algorithm = algorithm;

        const startTime = performance.now();
        let timeoutId = null;

        try {
            const pathfinderPromise = new Promise((resolve, reject) => {
                if (algorithm.includes('ml')) {
                    this.pathfinder.runAsync(start, end, null, (success, nodes, dist) => {
                        resolve({ success, nodes, dist });
                    });
                } else {
                    const pfResult = this.pathfinder.run(start, end);
                    resolve(pfResult);
                }
            });

            const timeoutPromise = new Promise((_, reject) => {
                timeoutId = setTimeout(() => reject(new Error('Timeout')), this.config.timeoutMs);
            });

            const { success, nodes, dist } = await Promise.race([pathfinderPromise, timeoutPromise]);

            result.success = success ? 1 : 0;
            result.path_cost = success ? dist : null;
            result.nodes_expanded = nodes;
            result.runtime_ms = performance.now() - startTime;
        } catch (error) {
            result.success = 0;
            result.path_cost = null;
            result.nodes_expanded = 0;
            result.runtime_ms = this.config.timeoutMs;
        } finally {
            if (timeoutId) clearTimeout(timeoutId);
        }

        return result;
    }

    exportToCsv() {
        const header = BatchTestResult.toCsvHeader();
        const rows = this.results.map(result => result.toCsvRow());
        return [header, ...rows].join('\n');
    }
}
