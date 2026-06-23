param(
    [string]$Python = "$PSScriptRoot\..\.venv\Scripts\python.exe",
    [string]$PaddleOcrRoot = "D:\paddleocr-2.7",
    [string]$DatasetRoot = "D:\echora_ocr_textocr_en",
    [string]$OutputRoot = "D:\echora_ocr_runs\ppocrv4_textocr_en"
)

$ErrorActionPreference = "Stop"
$weightsDir = Join-Path $PaddleOcrRoot "pretrained"
$weightsBase = Join-Path $weightsDir "en_PP-OCRv4_rec_train\best_accuracy"
$weightsArchive = Join-Path $weightsDir "en_PP-OCRv4_rec_train.tar"
$weightsUrl = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar"

foreach ($path in @($Python, (Join-Path $PaddleOcrRoot "tools\train.py"),
                         (Join-Path $DatasetRoot "train_list.txt"),
                         (Join-Path $DatasetRoot "val_list.txt"))) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

if (-not (Test-Path -LiteralPath "${weightsBase}.pdparams")) {
    New-Item -ItemType Directory -Path $weightsDir -Force | Out-Null
    Write-Host "Downloading official English PP-OCRv4 training weights..."
    Invoke-WebRequest -Uri $weightsUrl -OutFile $weightsArchive
    tar.exe -xf $weightsArchive -C $weightsDir
}

Push-Location $PaddleOcrRoot
try {
    Write-Host "Installing PaddleOCR training requirements..."
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "PaddleOCR dependency installation failed." }

    # PaddleOCR 2.7 uses OpenCV wheels built against NumPy 1.x.
    & $Python -m pip install --force-reinstall "numpy==1.26.4"
    if ($LASTEXITCODE -ne 0) { throw "Compatible NumPy installation failed." }

    & $Python -m pip install "nvidia-cudnn-cu11==8.9.5.29"
    if ($LASTEXITCODE -ne 0) { throw "NVIDIA cuDNN installation failed." }

    $environmentRoot = Split-Path (Split-Path $Python -Parent) -Parent
    $cudnnBin = Join-Path $environmentRoot "Lib\site-packages\nvidia\cudnn\bin"
    $cublasBin = Join-Path $environmentRoot "Lib\site-packages\nvidia\cublas\bin"
    $nvrtcBin = Join-Path $environmentRoot "Lib\site-packages\nvidia\cuda_nvrtc\bin"
    foreach ($dllDirectory in @($cudnnBin, $cublasBin, $nvrtcBin)) {
        if (-not (Test-Path -LiteralPath $dllDirectory)) {
            throw "NVIDIA DLL directory was not found: $dllDirectory"
        }
    }
    $env:PATH = "$cudnnBin;$cublasBin;$nvrtcBin;$env:PATH"

    & $Python -c "import paddle, sys; gpu=paddle.device.is_compiled_with_cuda(); print('Paddle:', paddle.__version__); print('GPU:', gpu); sys.exit(0 if gpu else 1)"
    if ($LASTEXITCODE -ne 0) { throw "A CUDA-enabled PaddlePaddle build is required for this GPU fine-tune." }

    $trainArgs = @(
        "tools/train.py",
        "-c", "configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
        "-o",
        "Global.pretrained_model=$weightsBase",
        "Global.save_model_dir=$OutputRoot",
        "Global.epoch_num=15",
        "Optimizer.lr.learning_rate=0.0001",
        "Optimizer.lr.warmup_epoch=1",
        "Train.dataset.data_dir=$DatasetRoot",
        "Train.dataset.label_file_list=['$DatasetRoot/train_list.txt']",
        "Train.sampler.first_bs=64",
        "Train.loader.num_workers=4",
        "Eval.dataset.data_dir=$DatasetRoot",
        "Eval.dataset.label_file_list=['$DatasetRoot/val_list.txt']",
        "Eval.loader.num_workers=2",
        "Global.use_gpu=True"
    )

    Write-Host "Starting TextOCR English fine-tuning..."
    & $Python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Fine-tuning failed." }
}
finally {
    Pop-Location
}
