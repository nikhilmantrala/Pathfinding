#!/usr/bin/env powershell
# Quick Start Guide for Using Improved ML Heuristic Model

param(
    [ValidateSet("test", "deploy", "compare", "backup", "restore", "info")]
    [string]$Action = "info"
)

function Write-Header {
    param([string]$Text)
    Write-Host "`n" -NoNewline
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Yellow
    Write-Host "=" * 60 -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Write-Error2 {
    param([string]$Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Write-Info {
    param([string]$Text)
    Write-Host "ℹ $Text" -ForegroundColor Blue
}

# Main actions
function Show-Info {
    Write-Header "IMPROVED ML HEURISTIC MODEL - STATUS"
    
    Write-Host "`n📊 Model Information:"
    Write-Info "Original Model: web_model_static/ (4-feature)"
    Write-Success "Improved Model: web_model_improved_static/ (5-feature)"
    
    Write-Host "`n📈 Key Differences:"
    Write-Host "  Feature 1-4: Normalized coordinates (start_r, start_c, goal_r, goal_c)"
    Write-Host "  Feature 5:   Normalized octile distance (NEW!)"
    Write-Host "  Size:        Original: $($(Get-ChildItem web_model_static -Recurse | Measure-Object -Property Length -Sum).Sum / 1KB) KB | Improved: $($(Get-ChildItem web_model_improved_static -Recurse | Measure-Object -Property Length -Sum).Sum / 1KB) KB"
    
    Write-Host "`n🧪 Test Pages Available:"
    Write-Info "test-model-improved.html - Comprehensive model testing"
    Write-Info "test-model-simple.html - Basic functionality test"
    
    Write-Host "`n📚 Documentation:"
    Write-Info "MODEL_CONVERSION_GUIDE.md - Full details and troubleshooting"
    
    Write-Host "`n🚀 Quick Commands:"
    Write-Host "  .\quickstart.ps1 test      - Test improved model"
    Write-Host "  .\quickstart.ps1 deploy    - Deploy improved model"
    Write-Host "  .\quickstart.ps1 backup    - Backup original model"
    Write-Host "  .\quickstart.ps1 restore   - Restore original model"
    Write-Host "  .\quickstart.ps1 compare   - Compare both models"
}

function Start-Test {
    Write-Header "TESTING IMPROVED MODEL"
    
    # Check if HTTP server is running
    $serverRunning = $false
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/test-model-improved.html" -TimeoutSec 2 -ErrorAction SilentlyContinue
        $serverRunning = $response.StatusCode -eq 200
    } catch {
        $serverRunning = $false
    }
    
    if (-not $serverRunning) {
        Write-Error2 "HTTP server not running!"
        Write-Info "Starting HTTP server on port 8000..."
        Start-Process python -ArgumentList "-m http.server 8000 --directory ." -WindowStyle Hidden
        Start-Sleep -Seconds 2
    }
    
    Write-Success "Opening test page in browser..."
    Write-Info "URL: http://localhost:8000/test-model-improved.html"
    
    try {
        Start-Process "http://localhost:8000/test-model-improved.html"
    } catch {
        Write-Error2 "Could not open browser automatically. Open manually:"
        Write-Host "  http://localhost:8000/test-model-improved.html"
    }
    
    Write-Host "`n📝 Next Steps:"
    Write-Host "  1. Click 'Load Improved Model (5-feat)'"
    Write-Host "  2. Click 'Test Model' to run single test"
    Write-Host "  3. Click 'Test 10 Random Grids' for batch testing"
    Write-Host "  4. Check console for predictions and diagnostics"
}

function Deploy-Model {
    Write-Header "DEPLOYING IMPROVED MODEL"
    
    # Check if web_model_static exists
    if (-not (Test-Path "web_model_static")) {
        Write-Error2 "web_model_static directory not found!"
        return
    }
    
    # Backup if not already backed up
    if (-not (Test-Path "web_model_static_backup")) {
        Write-Info "Creating backup of original model..."
        Copy-Item -Path "web_model_static" -Destination "web_model_static_backup" -Recurse
        Write-Success "Backup created: web_model_static_backup/"
    } else {
        Write-Info "Backup already exists: web_model_static_backup/"
    }
    
    # Replace with improved model
    Write-Info "Replacing original model with improved model..."
    Remove-Item -Path "web_model_static" -Recurse -Force
    Copy-Item -Path "web_model_improved_static" -Destination "web_model_static" -Recurse
    
    Write-Success "Improved model deployed!"
    Write-Info "ml-heuristic.js will now use: web_model_static/ (improved 5-feature model)"
    
    Write-Host "`n⚠️  Verify in ml-heuristic.js that it's using [1, 5] tensors:"
    Write-Host "  const sgTensor = tf.tensor2d([sg], [1, 5], 'float32');"
}

function Backup-Model {
    Write-Header "BACKING UP ORIGINAL MODEL"
    
    if (-not (Test-Path "web_model_static")) {
        Write-Error2 "web_model_static directory not found!"
        return
    }
    
    if (Test-Path "web_model_static_backup") {
        Write-Info "Backup already exists. Overwriting..."
    }
    
    Copy-Item -Path "web_model_static" -Destination "web_model_static_backup" -Recurse -Force
    Write-Success "Backup created: web_model_static_backup/"
    Write-Host "  Original model is safe. You can deploy the improved model."
}

function Restore-Model {
    Write-Header "RESTORING ORIGINAL MODEL"
    
    if (-not (Test-Path "web_model_static_backup")) {
        Write-Error2 "Backup not found! Cannot restore."
        Write-Host "  If you deployed the improved model, you can:"
        Write-Host "  1. Copy from web_model_improved_static back to web_model_static"
        Write-Host "  2. Restore from version control"
        return
    }
    
    Write-Info "Restoring original model..."
    Remove-Item -Path "web_model_static" -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item -Path "web_model_static_backup" -Destination "web_model_static" -Recurse
    
    Write-Success "Original model restored!"
    Write-Info "ml-heuristic.js is now using: web_model_static/ (original 4-feature model)"
}

function Compare-Models {
    Write-Header "COMPARING MODELS"
    
    Write-Host "`n📋 Original Model (web_model_static):"
    if (Test-Path "web_model_static") {
        $size = (Get-ChildItem "web_model_static" -Recurse | Measure-Object -Property Length -Sum).Sum / 1KB
        Write-Host "  Status: Present"
        Write-Host "  Size: $([math]::Round($size, 1)) KB"
        Write-Host "  Features: 4 (start_r, start_c, goal_r, goal_c)"
    } else {
        Write-Host "  Status: Not found (may have been replaced)"
    }
    
    Write-Host "`n📋 Improved Model (web_model_improved_static):"
    if (Test-Path "web_model_improved_static") {
        $size = (Get-ChildItem "web_model_improved_static" -Recurse | Measure-Object -Property Length -Sum).Sum / 1KB
        Write-Success "Present and ready"
        Write-Host "  Size: $([math]::Round($size, 1)) KB"
        Write-Host "  Features: 5 (+ normalized distance)"
    } else {
        Write-Error2 "Not found!"
    }
    
    Write-Host "`n⚖️  Comparison:"
    Write-Host "  Original:  4-feature model, baseline performance"
    Write-Host "  Improved:  5-feature model, better distance sensitivity"
    
    Write-Host "`n💡 Recommendation: Deploy improved model and test!"
}

# Route to appropriate action
switch ($Action) {
    "test" { Start-Test }
    "deploy" { Deploy-Model }
    "backup" { Backup-Model }
    "restore" { Restore-Model }
    "compare" { Compare-Models }
    default { Show-Info }
}

Write-Host "`n"
