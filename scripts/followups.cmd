@echo off
rem Follow-up queue for docs/followups.md. Run from the repository root.
rem Set PYTHON, ADAPTER_BITS_OUT and ADAPTER_BITS_HF_CACHE (and HF_HUB_OFFLINE=1) first.
rem Jobs run one at a time; each waits for 12 GB of free commit before loading the model.
if "%PYTHON%"=="" set PYTHON=python
set OUT=%ADAPTER_BITS_OUT%
if "%OUT%"=="" set OUT=runs
set GD27=--plan lora:gate:1:256 --optimizer gd --batch 4096 --pool 4096 --cache-file %OUT%\cache-arrow-seed0-4096.pt --hessian-items 256 --hessian-iters 10 --grad-items 256
set GD24=--first 24 --pool 256 --items 256 --specs 24:4 --optimizer gd --batch 256 --hessian-items 256 --hessian-iters 10 --grad-items 256

rem A1: layer 27, full-batch gradient descent at equal gradient-flow time T = eta * steps = 400
%PYTHON% -u src\adapter_bits.py %GD27% --lora-lr 0.1 --steps 4000 --eval-every 250 --sharpness-every 2 --name fu-a1-eta0.1 > %OUT%\fu-a1-eta0.1.log 2>&1
%PYTHON% -u src\adapter_bits.py %GD27% --lora-lr 0.03 --steps 13333 --eval-every 1000 --sharpness-every 1 --name fu-a1-eta0.03 > %OUT%\fu-a1-eta0.03.log 2>&1
%PYTHON% -u src\adapter_bits.py %GD27% --lora-lr 0.01 --steps 40000 --eval-every 2000 --sharpness-every 1 --name fu-a1-eta0.01 > %OUT%\fu-a1-eta0.01.log 2>&1
%PYTHON% -u src\adapter_bits.py %GD27% --lora-lr 0.003 --steps 133333 --eval-every 5000 --sharpness-every 1 --name fu-a1-eta0.003 > %OUT%\fu-a1-eta0.003.log 2>&1

rem B1: seeds 1 and 2, single layer, Adam
%PYTHON% -u src\adapter_bits.py --seed 1 --pool 4096 --cache-file %OUT%\cache-arrow-seed1-4096.pt --plan probe:64:128,probe:512:1024,lora:gate:1:256,lora:gate:8:512 --probe-lr 0.05 --lora-lr 0.03 --steps 3000 --name fu-b1-seed1 > %OUT%\fu-b1-seed1.log 2>&1
%PYTHON% -u src\adapter_bits.py --seed 2 --pool 4096 --cache-file %OUT%\cache-arrow-seed2-4096.pt --plan probe:64:128,probe:512:1024,lora:gate:1:256,lora:gate:8:512 --probe-lr 0.05 --lora-lr 0.03 --steps 3000 --name fu-b1-seed2 > %OUT%\fu-b1-seed2.log 2>&1

rem B2: seeds 1 and 2, placement and spread at W = 16,384, Adam
%PYTHON% -u src\adapter_bits_deep.py --seed 1 --first 24 --pool 2048 --items 2048 --specs "27:4;24:4;24:1,25:1,26:1,27:1" --lr 0.03 --steps 4000 --grad-items 512 --name fu-b2-seed1 > %OUT%\fu-b2-seed1.log 2>&1
%PYTHON% -u src\adapter_bits_deep.py --seed 2 --first 24 --pool 2048 --items 2048 --specs "27:4;24:4;24:1,25:1,26:1,27:1" --lr 0.03 --steps 4000 --grad-items 512 --name fu-b2-seed2 > %OUT%\fu-b2-seed2.log 2>&1

rem A2: layer 24 alone, full-batch gradient descent at T = 400
%PYTHON% -u src\adapter_bits_deep.py %GD24% --lr 0.1 --steps 4000 --eval-every 250 --sharpness-every 2 --name fu-a2-eta0.1 > %OUT%\fu-a2-eta0.1.log 2>&1
%PYTHON% -u src\adapter_bits_deep.py %GD24% --lr 0.03 --steps 13333 --eval-every 1000 --sharpness-every 1 --name fu-a2-eta0.03 > %OUT%\fu-a2-eta0.03.log 2>&1

rem C: four-layer adapter at M = 16,384, 12,000-step cap, rank measured on 2,048 items
%PYTHON% -u src\adapter_bits_deep.py --first 24 --pool 16384 --items 16384 --specs 24:1,25:1,26:1,27:1 --lr 0.03 --steps 12000 --grad-items 2048 --name fu-c-sat > %OUT%\fu-c-sat.log 2>&1

echo followups-done > %OUT%\followups.done
