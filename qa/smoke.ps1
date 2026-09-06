#Requires -Version 7.0
<#
.SYNOPSIS
    T14: Live QA 驱动脚本 — 启动 mock 服务与真实 ComfyUI(CPU), 跑 S1/S3(REST)/S4/S5 场景并产出证据。
.DESCRIPTION
    用法:
      pwsh -NoProfile -File qa/smoke.ps1 -DryRun   # 只打印计划 + 校验环境, 不做任何变更, exit 0
      pwsh -NoProfile -File qa/smoke.ps1           # 真实执行 (由 T15 在人工监督下运行)
    退出码: 0 = 全部场景 PASS; 1 = 任一场景 FAIL 或流程中断。
    保证:
      - teardown 永远执行 (try/finally): 杀进程、等端口释放、删除本次创建的 junction (仅 rmdir, 绝不 /S)、清缓存目录。
      - evidence 目录只增不删。
      - S3 的 DOM 半边由编排器 (Playwright) 负责, 不在本脚本范围。
#>
param(
    [switch]$DryRun,            # 打印计划动作与解析后的路径, 不做任何变更, exit 0
    [int]$MockPort = 18099,     # mock 上游服务端口
    [int]$ComfyPort = 18188,    # ComfyUI 服务端口
    [string]$ComfyDir = "E:\Work\project\Python\ComfyUI",
    [string]$PluginDir = "E:\Work\project\Python\ComfyUI-OpenAPI"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ---------- 路径集中解析 ----------
$EvidenceDir  = Join-Path $PluginDir "qa\evidence"
$JunctionPath = Join-Path $ComfyDir "custom_nodes\ComfyUI-OpenAPI"
$CacheDir     = Join-Path $env:TEMP "ulw-qa-cache"
$WfDir        = Join-Path $PluginDir "qa\workflows"
$MockScript   = Join-Path $WfDir "..\run_mock_server.py"   # 即 qa\run_mock_server.py
$S1Json       = Join-Path $WfDir "s1_ok.json"
$S4AuthJson   = Join-Path $WfDir "s4_auth_error.json"
$S4ParamsJson = Join-Path $WfDir "s4_bad_params.json"
$S3Json       = Join-Path $WfDir "s3_linked_prompt.json"
$S6Json       = Join-Path $WfDir "s6_dashscope_ok.json"
$S7Json       = Join-Path $WfDir "s7_dashscope_404.json"
$MockLog      = Join-Path $EvidenceDir "mock_stdout.log"
$ComfyOutLog  = Join-Path $EvidenceDir "comfy_stdout.log"
$ComfyErrLog  = Join-Path $EvidenceDir "comfy_stderr.log"

# ---------- 运行状态跟踪 (teardown 依赖) ----------
$script:MockPID = $null          # mock 进程 PID
$script:ComfyPID = $null         # ComfyUI 进程 PID
$script:JunctionCreated = $false # junction 是否由本次运行创建 (只有本次创建的才删除)
$script:LogFile = $null          # DryRun 下为 null, 不写文件
$script:FailureNotes = New-Object System.Collections.Generic.List[string]
# 场景结果: 固定键序, 与 s0_summary.json 结构一致; 初值全 false, 只有显式判定 PASS 才置 true
$script:Results = [ordered]@{ s1 = $false; s3_rest = $false; s4_auth = $false; s4_params = $false; s5 = $false; s6_dashscope = $false; s7_dashscope_err = $false }

# ---------- 工具函数 ----------

# 统一日志: 控制台 + (真实运行时) s0_run.log
function Write-Log {
    param([string]$Msg)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Msg
    Write-Host $line
    if ($script:LogFile) { Add-Content -LiteralPath $script:LogFile -Value $line -ErrorAction SilentlyContinue }
}

# 记录场景判定: PASS/FAIL + 原因, 同时落一份 <场景>_result.txt 证据
function Set-ScenarioResult {
    param([string]$Name, [bool]$Passed, [string]$Reason)
    $script:Results[$Name] = $Passed
    $verdict = if ($Passed) { "PASS" } else { "FAIL" }
    Write-Log ("{0}: {1} - {2}" -f $Name, $verdict, $Reason)
    if (-not $Passed) { $script:FailureNotes.Add("$verdict $Name : $Reason") }
    Set-Content -LiteralPath (Join-Path $EvidenceDir "${Name}_result.txt") -Value "${verdict} ${Reason}"
}

# StrictMode 安全取属性: 不存在返回 $null (支持 PSCustomObject 与字典)
function Get-Prop {
    param($Obj, [string]$Name)
    if ($null -eq $Obj) { return $null }
    if ($Obj -is [System.Collections.IDictionary]) {
        if ($Obj.Contains($Name)) { return $Obj[$Name] } else { return $null }
    }
    $p = $Obj.PSObject.Properties[$Name]
    if ($p) { return $p.Value } else { return $null }
}

# 等待端口释放 (最多 $TimeoutSec 秒)
function Wait-PortFree {
    param([int]$Port, [int]$TimeoutSec = 10)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $c) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# STEP 4: workflow 执行器 — POST /prompt 后轮询 /history/<prompt_id> (2s 间隔, ≤60s)
# 返回 history 条目 (PSCustomObject) 或 $null; 原始 history 落盘到 <EvidenceBase>.json
function Run-Workflow {
    param([string]$JsonPath, [string]$EvidenceBase)
    $text = Get-Content -LiteralPath $JsonPath -Raw
    $status = 0
    $resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$ComfyPort/prompt" -ContentType "application/json" -Body $text -TimeoutSec 30 -SkipHttpErrorCheck -StatusCodeVariable status
    if ($status -ne 200) {
        # 校验失败的响应体直接进失败原因, 便于定位 (如 COMBO 值不在当前缓存选项列表)
        throw "POST /prompt 返回 HTTP ${status}: $(($resp | ConvertTo-Json -Depth 8 -Compress))"
    }
    $promptId = Get-Prop $resp "prompt_id"
    if (-not $promptId) { throw "POST /prompt 响应缺少 prompt_id" }
    Write-Log "  prompt_id=$promptId, 轮询 /history (最长 60s)"
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$ComfyPort/history/$promptId" -TimeoutSec 10
            $entry = Get-Prop $h $promptId
            if ($entry) {
                # 保存原始 history JSON 作为证据
                $h | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$EvidenceBase.json"
                return $entry
            }
        } catch {
            # 执行完成前 history 端点可能瞬时不可达/404: 吞掉, 继续轮询
        }
        Start-Sleep -Seconds 2
    }
    return $null
}

# ---------- DryRun: 只打印计划与校验环境, 不做任何变更 ----------
if ($DryRun) {
    Write-Host "=== SMOKE DRY RUN (不创建目录/不建 junction/不启动进程/不绑定端口) ==="
    # 环境校验
    if (-not (Test-Path -LiteralPath $ComfyDir))  { Write-Host "DRYRUN_FAIL: ComfyDir 不存在: $ComfyDir"; exit 1 }
    if (-not (Test-Path -LiteralPath $PluginDir)) { Write-Host "DRYRUN_FAIL: PluginDir 不存在: $PluginDir"; exit 1 }
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { Write-Host "DRYRUN_FAIL: PATH 中找不到 python"; exit 1 }
    Write-Host "环境校验 OK: ComfyDir/PluginDir 存在; python -> $($py.Source)"
    # 打印解析后的路径
    Write-Host "--- 解析后的路径 ---"
    Write-Host "EvidenceDir  = $EvidenceDir"
    Write-Host "JunctionPath = $JunctionPath"
    Write-Host "CacheDir     = $CacheDir"
    Write-Host "MockScript   = $((Resolve-Path -LiteralPath $MockScript -ErrorAction SilentlyContinue) ?? $MockScript)"
    foreach ($wf in @($S1Json, $S4AuthJson, $S4ParamsJson, $S3Json)) {
        # workflow 文件与脚本并行创建; 缺失只警告, 不算 DryRun 失败
        $mark = if (Test-Path -LiteralPath $wf) { "OK" } else { "WARNING: 尚未生成 (并行任务负责)" }
        Write-Host "Workflow     = $wf  [$mark]"
    }
    # 打印将要执行的命令
    Write-Host "--- 将要执行的命令 ---"
    Write-Host '[STEP0] New-Item -Directory (evidence)'
    Write-Host "[STEP0] if (-not Test-Path `"$JunctionPath`") cmd /c `"mklink /J `"$JunctionPath`" `"$PluginDir`"`""
    Write-Host "[STEP1] `$env:COMFYUI_OPENAPI_CACHE_DIR = `"$CacheDir`"; New-Item -Directory `$CacheDir"
    Write-Host "[STEP1] Start-Process python -ArgumentList 'qa/run_mock_server.py','--port','$MockPort' -WorkingDirectory `"$PluginDir`" -> $MockLog (等待 MOCK_READY ≤30s)"
    Write-Host "[STEP2] Start-Process python -ArgumentList 'main.py','--cpu','--port','$ComfyPort','--listen','127.0.0.1','--dont-print-server' -WorkingDirectory `"$ComfyDir`" -> $ComfyOutLog / $ComfyErrLog (轮询 /system_stats ≤120s)"
    Write-Host "[STEP3] tail -60 两份 comfy 日志 -> s5_startup.log; GET /object_info -> s5_object_info_keys.txt"
    Write-Host "[STEP4.5] POST /api/comfyui_openapi/fetch_models (缓存预热, 期望 7 模型) -> s2_fetch_models_route.json"
    Write-Host "[STEP5] POST /prompt $S1Json -> s1_history.json; GET /view -> s1_view.png (>1000B); GET ${MockPort}/debug/last_request -> s1_last_request.json"
    Write-Host "[STEP6] POST /prompt $S4AuthJson -> s4_auth.json; POST /prompt $S4ParamsJson -> s4_params.json"
    Write-Host "[STEP7] POST /prompt $S3Json -> s3_workflow.json; GET /debug/last_request -> s3_last_request.json"
    Write-Host "[STEP7.5] POST fetch_models(protocol=dashscope) 目录路由 -> s6_catalog_route.json; POST /prompt $S6Json -> s6_history.json + /debug/last_native_request -> s6_last_native_request.json"
    Write-Host "[STEP7.6] POST /prompt $S7Json -> s7_history.json (断言错误含 'Model not exist')"
    Write-Host "[STEP8] 写 s0_summary.json; 打印 SMOKE_PASS / SMOKE_FAIL"
    Write-Host "[TEARDOWN] Stop-Process(mock,comfy); 等端口释放 ≤10s; 若本次创建则 cmd /c `"rmdir `"$JunctionPath`"`" (绝不 /S); Remove-Item `$CacheDir; 保留 evidence"
    Write-Host "=== DRY RUN 结束: 未做任何变更 ==="
    exit 0
}

# ---------- 真实执行 ----------
# 创建 evidence 目录 (存在则不动, 绝不删除其中内容)
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$script:LogFile = Join-Path $EvidenceDir "s0_run.log"
Write-Log "smoke.ps1 开始执行 (MockPort=$MockPort ComfyPort=$ComfyPort)"

try {
    try {
        # ---- STEP 0: 插件 junction (幂等: 已存在则跳过创建但仍校验类型) ----
        Write-Log "STEP 0: 校验/创建插件 junction"
        if (Test-Path -LiteralPath $JunctionPath) {
            Write-Log "  junction 已存在, 跳过创建"
        } else {
            cmd /c "mklink /J `"$JunctionPath`" `"$PluginDir`""
            $script:JunctionCreated = $true
            Write-Log "  已创建 junction (本次运行负责回收)"
        }
        $ji = Get-Item -LiteralPath $JunctionPath -Force
        if ($ji.LinkType -ne "Junction") { throw "STEP 0 失败: $JunctionPath LinkType=$($ji.LinkType), 期望 Junction" }

        # ---- STEP 1: 启动 mock 服务 ----
        Write-Log "STEP 1: 启动 mock (端口 $MockPort)"
        $env:COMFYUI_OPENAPI_CACHE_DIR = $CacheDir
        New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
        $mockProc = Start-Process -FilePath "python" `
            -ArgumentList @("qa/run_mock_server.py", "--port", "$MockPort") `
            -WorkingDirectory $PluginDir -PassThru -RedirectStandardOutput $MockLog
        $script:MockPID = $mockProc.Id
        Write-Log "  mock PID=$($script:MockPID), 轮询 MOCK_READY (≤30s)"
        $mockReady = $false
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            if ($mockProc.HasExited) { break }
            $content = Get-Content -LiteralPath $MockLog -Raw -ErrorAction SilentlyContinue
            if ($content -and $content.Contains("MOCK_READY")) { $mockReady = $true; break }
            Start-Sleep -Milliseconds 500
        }
        if (-not $mockReady) {
            if ($mockProc.HasExited) { throw "mock not ready: 进程已退出 ExitCode=$($mockProc.ExitCode)" }
            throw "mock not ready: 30s 内 mock_stdout.log 未出现 MOCK_READY"
        }
        Write-Log "  mock 就绪"

        # ---- STEP 2: 启动 ComfyUI (CPU) ----
        Write-Log "STEP 2: 启动 ComfyUI (端口 $ComfyPort, --cpu)"
        $comfyProc = Start-Process -FilePath "python" `
            -ArgumentList @("main.py", "--cpu", "--port", "$ComfyPort", "--listen", "127.0.0.1", "--dont-print-server") `
            -WorkingDirectory $ComfyDir -PassThru `
            -RedirectStandardOutput $ComfyOutLog -RedirectStandardError $ComfyErrLog
        $script:ComfyPID = $comfyProc.Id
        Write-Log "  comfy PID=$($script:ComfyPID), 轮询 /system_stats (2s 间隔, ≤120s)"
        $comfyReady = $false
        $deadline = (Get-Date).AddSeconds(120)
        while ((Get-Date) -lt $deadline) {
            if ($comfyProc.HasExited) { break }
            try {
                $null = Invoke-RestMethod -Uri "http://127.0.0.1:$ComfyPort/system_stats" -TimeoutSec 5
                $comfyReady = $true
                break
            } catch {
                Start-Sleep -Seconds 2
            }
        }
        if (-not $comfyReady) {
            if ($comfyProc.HasExited) { throw "ComfyUI 进程提前退出 ExitCode=$($comfyProc.ExitCode), 见 comfy_stderr.log" }
            throw "ComfyUI 未就绪: 120s 内 /system_stats 不可达"
        }
        Write-Log "  ComfyUI 就绪"

        # ---- STEP 3: S5 证据 (启动日志 + object_info) ----
        Write-Log "STEP 3: S5 证据采集"
        $s5Log = Join-Path $EvidenceDir "s5_startup.log"
        $tail = @()
        $tail += @(Get-Content -LiteralPath $ComfyOutLog -Tail 60 -ErrorAction SilentlyContinue)
        $tail += @(Get-Content -LiteralPath $ComfyErrLog -Tail 60 -ErrorAction SilentlyContinue)
        Set-Content -LiteralPath $s5Log -Value $tail

        $objInfo = Invoke-RestMethod -Uri "http://127.0.0.1:$ComfyPort/object_info" -TimeoutSec 60
        $keys = @($objInfo.PSObject.Properties.Name | Sort-Object)
        Set-Content -LiteralPath (Join-Path $EvidenceDir "s5_object_info_keys.txt") -Value $keys

        $s5Reasons = @()
        foreach ($need in @("OpenAPIImageGenerator", "PreviewImage", "CheckpointLoaderSimple")) {
            if ($keys -notcontains $need) { $s5Reasons += "object_info 缺少节点 $need" }
        }
        # Traceback 判定: 仅当 Traceback 后 5 行内提及 ComfyUI-OpenAPI 才算插件故障;
        # ComfyUI 核心其他位置的 Traceback 不算失败 (Select-String -Context 0,5 实现)
        foreach ($hit in @(Select-String -LiteralPath $s5Log -Pattern "Traceback" -Context 0,5 -ErrorAction SilentlyContinue)) {
            $near = @($hit.Line) + @($hit.Context.PostContext)
            if (($near -join "`n") -match "ComfyUI-OpenAPI") {
                $s5Reasons += "启动日志第 $($hit.LineNumber) 行附近的 Traceback 涉及 ComfyUI-OpenAPI"
                break
            }
        }
        Set-ScenarioResult "s5" ($s5Reasons.Count -eq 0) $(
            if ($s5Reasons.Count) { $s5Reasons -join "; " } else { "三节点均已注册, 无涉及插件的启动 Traceback" })

        # ---- STEP 4.5: 缓存预热 (S2 后端实机证明) ----
        # 真实用户流程是「先点『获取模型列表』把远端模型写入本地缓存, 再选模型运行」。
        # /prompt 校验按当前缓存求值 model COMBO (空缓存只有哨兵项, 任意模型值都会被
        # ComfyUI 校验拒绝 → 400), 因此必须先调用插件自身的 fetch_models 路由完成预热。
        # 该调用同时就是 S2 场景的后端实机证据: 代理路由 + 远端拉取 + 缓存持久化。
        Write-Log "STEP 4.5: 缓存预热 (POST /api/comfyui_openapi/fetch_models)"
        $warmBody = @{ base_url = "http://127.0.0.1:$MockPort"; api_key = "qa-test-key" } | ConvertTo-Json -Compress
        $warmStatus = 0
        $warm = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$ComfyPort/api/comfyui_openapi/fetch_models" -ContentType "application/json" -Body $warmBody -TimeoutSec 60 -SkipHttpErrorCheck -StatusCodeVariable warmStatus
        $warm | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $EvidenceDir "s2_fetch_models_route.json")
        if ($warmStatus -ne 200 -or -not (Get-Prop $warm "ok")) {
            throw "缓存预热失败: HTTP $warmStatus, 响应: $(($warm | ConvertTo-Json -Depth 6 -Compress))"
        }
        $warmModels = @(Get-Prop $warm "models")
        if ($warmModels.Count -ne 7) { throw "缓存预热异常: 期望 7 个模型, 实际 $($warmModels.Count)" }
        Write-Log "  预热完成: $($warmModels.Count) 个模型已入缓存 ($($warmModels -join ', '))"

        # ---- STEP 5: S1 正常出图 ----
        Write-Log "STEP 5: S1 (s1_ok.json 正常出图链路)"
        try {
            $entry = Run-Workflow -JsonPath $S1Json -EvidenceBase (Join-Path $EvidenceDir "s1_history")
            if (-not $entry) { throw "60s 内 history 无该 prompt_id 条目" }
            $imgs = Get-Prop (Get-Prop (Get-Prop $entry "outputs") "2") "images"
            if (-not $imgs -or @($imgs).Count -eq 0) { throw "outputs.'2'.images 缺失或为空" }
            $img = @($imgs)[0]
            $fn = Get-Prop $img "filename"
            if (-not $fn) { throw "images[0].filename 为空" }
            # /view 查询串完全由 images[0] 的 filename/subfolder/type 字段构造
            $imgType = (Get-Prop $img "type") ?? "output"
            $q = "filename={0}&type={1}" -f [uri]::EscapeDataString($fn), [uri]::EscapeDataString($imgType)
            $sub = Get-Prop $img "subfolder"
            if ($sub) { $q += "&subfolder=" + [uri]::EscapeDataString($sub) }
            $viewPath = Join-Path $EvidenceDir "s1_view.png"
            Invoke-WebRequest -Uri "http://127.0.0.1:$ComfyPort/view?$q" -OutFile $viewPath -TimeoutSec 30 | Out-Null
            $viewLen = (Get-Item -LiteralPath $viewPath).Length
            # mock 端无鉴权调试端点: 校验插件真正发出的请求体 (证据先落盘, 再做判定)
            # 注意: /debug/last_request 直接返回请求体本身, 无 body 包装层
            $last = Invoke-RestMethod -Uri "http://127.0.0.1:$MockPort/debug/last_request" -TimeoutSec 10
            $last | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $EvidenceDir "s1_last_request.json")
            # 图像有效性判定: PNG 魔数 + 最小字节数。纯色 PNG 压缩率极高
            # (64x64 纯红仅 ~500 字节), 字节阈值必须放低, 以魔数为准。
            $pngMagic = [byte[]](0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
            $head = [System.IO.File]::ReadAllBytes($viewPath)
            $isPng = ($head.Length -ge 8) -and [System.Linq.Enumerable]::SequenceEqual([byte[]]$head[0..7], $pngMagic)
            if (-not $isPng -or $viewLen -le 100) { throw "s1_view.png 不是有效 PNG 或过小 ($viewLen 字节)" }
            if ((Get-Prop $last "prompt") -ne "a red square") { throw "last_request prompt 不为 'a red square' (实际: $(Get-Prop $last 'prompt'))" }
            if ((Get-Prop $last "model") -ne "mock-b64") { throw "last_request model 不为 'mock-b64' (实际: $(Get-Prop $last 'model'))" }
            Set-ScenarioResult "s1" $true "出图文件名=$fn; /view 下载有效 PNG $viewLen 字节; mock 收到正确请求体"
        } catch {
            Set-ScenarioResult "s1" $false $_.Exception.Message
        }

        # ---- STEP 6: S4 错误传播 (鉴权失败 / 非法参数) ----
        Write-Log "STEP 6: S4 错误场景"
        try {
            $entry = Run-Workflow -JsonPath $S4AuthJson -EvidenceBase (Join-Path $EvidenceDir "s4_auth")
            if (-not $entry) { throw "history 无条目" }
            $ss = Get-Prop (Get-Prop $entry "status") "status_str"
            $json = $entry | ConvertTo-Json -Depth 10
            # ComfyUI history 对执行失败的真实取值是 "error" (非 "failed"), 已实机核实
            if ($ss -ne "error") { throw "status_str='$ss', 期望 'error'" }
            if ($json -notmatch "Invalid API key") { throw "history 序列化中不含 'Invalid API key'" }
            Set-ScenarioResult "s4_auth" $true "执行失败且错误信息含 'Invalid API key'"
        } catch {
            Set-ScenarioResult "s4_auth" $false $_.Exception.Message
        }
        try {
            $entry = Run-Workflow -JsonPath $S4ParamsJson -EvidenceBase (Join-Path $EvidenceDir "s4_params")
            if (-not $entry) { throw "history 无条目" }
            $ss = Get-Prop (Get-Prop $entry "status") "status_str"
            $json = $entry | ConvertTo-Json -Depth 10
            if ($ss -ne "error") { throw "status_str='$ss', 期望 'error'" }
            if ($json -notmatch "params") { throw "history 序列化中不含 'params'" }
            Set-ScenarioResult "s4_params" $true "执行失败且错误信息含 'params'"
        } catch {
            Set-ScenarioResult "s4_params" $false $_.Exception.Message
        }

        # ---- STEP 7: S3 REST 半边 (链路注入 prompt) ----
        # 注意: S3 的 DOM 半边由编排器经 Playwright 完成, 本脚本不负责
        Write-Log "STEP 7: S3 REST 半边 (s3_linked_prompt.json)"
        try {
            $entry = Run-Workflow -JsonPath $S3Json -EvidenceBase (Join-Path $EvidenceDir "s3_workflow")
            if (-not $entry) { throw "history 无条目" }
            $ss = Get-Prop (Get-Prop $entry "status") "status_str"
            if ($ss -ne "success") { throw "status_str='$ss', S3 要求执行完成 (success)" }
            $last = Invoke-RestMethod -Uri "http://127.0.0.1:$MockPort/debug/last_request" -TimeoutSec 10
            $last | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $EvidenceDir "s3_last_request.json")
            # /debug/last_request 直接返回请求体本身, 无 body 包装层
            $bp = Get-Prop $last "prompt"
            if ($bp -ne "linked-prompt-marker") { throw "last_request prompt 不为 'linked-prompt-marker' (实际: '$bp')" }
            Set-ScenarioResult "s3_rest" $true "执行完成且 mock 收到 linked-prompt-marker"
        } catch {
            Set-ScenarioResult "s3_rest" $false $_.Exception.Message
        }

        # ---- STEP 7.5: S6 DashScope 原生协议链路 (v0.2) ----
        Write-Log "STEP 7.5: S6 (目录路由 + s6_dashscope_ok.json 原生出图)"
        try {
            # (a) fetch_models 路由的 dashscope 分支: 返回内置目录并写入缓存 (无远端调用)
            $catBody = @{ base_url = "http://127.0.0.1:$MockPort"; api_key = "qa-test-key"; protocol = "dashscope" } | ConvertTo-Json -Compress
            $catStatus = 0
            $cat = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$ComfyPort/api/comfyui_openapi/fetch_models" -ContentType "application/json" -Body $catBody -TimeoutSec 30 -SkipHttpErrorCheck -StatusCodeVariable catStatus
            $cat | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $EvidenceDir "s6_catalog_route.json")
            if ($catStatus -ne 200 -or -not (Get-Prop $cat "ok")) { throw "目录路由失败: HTTP $catStatus $(($cat | ConvertTo-Json -Depth 4 -Compress))" }
            $catModels = @(Get-Prop $cat "models")
            if ($catModels.Count -lt 2 -or "qwen-image-3.0-pro" -notin $catModels) { throw "目录路由未返回 qwen-image 目录 (实际: $($catModels -join ', '))" }
            # (b) 原生协议出图工作流
            $entry = Run-Workflow -JsonPath $S6Json -EvidenceBase (Join-Path $EvidenceDir "s6_history")
            if (-not $entry) { throw "history 无条目" }
            $ss = Get-Prop (Get-Prop $entry "status") "status_str"
            if ($ss -ne "success") { throw "status_str='$ss', S6 要求 success" }
            $imgs = Get-Prop (Get-Prop (Get-Prop $entry "outputs") "2") "images"
            if (-not $imgs -or @($imgs).Count -eq 0) { throw "outputs.'2'.images 缺失或为空" }
            # (c) mock 收到的原生请求体: model / input.messages[].text / parameters 透传
            $native = Invoke-RestMethod -Uri "http://127.0.0.1:$MockPort/debug/last_native_request" -TimeoutSec 10
            $native | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $EvidenceDir "s6_last_native_request.json")
            if ((Get-Prop $native "model") -ne "qwen-image-3.0-pro") { throw "native 请求 model 不为 qwen-image-3.0-pro" }
            $text = $native.input.messages[0].content[0].text
            if ($text -ne "a red square") { throw "native input.messages text 不为 'a red square' (实际: '$text')" }
            if ($native.parameters.size -ne "64*64") { throw "parameters.size 未透传 (实际: $($native.parameters.size))" }
            Set-ScenarioResult "s6_dashscope" $true "目录路由($($catModels.Count)模型)+原生出图+请求体结构全部正确"
        } catch {
            Set-ScenarioResult "s6_dashscope" $false $_.Exception.Message
        }

        # ---- STEP 7.6: S7 DashScope 错误信封传播 (v0.2) ----
        Write-Log "STEP 7.6: S7 (s7_dashscope_404.json 错误传播)"
        try {
            $entry = Run-Workflow -JsonPath $S7Json -EvidenceBase (Join-Path $EvidenceDir "s7_history")
            if (-not $entry) { throw "history 无条目" }
            $ss = Get-Prop (Get-Prop $entry "status") "status_str"
            if ($ss -ne "error") { throw "status_str='$ss', S7 要求 error" }
            $json = $entry | ConvertTo-Json -Depth 10
            if ($json -notmatch "Model not exist") { throw "history 序列化中不含 'Model not exist'" }
            Set-ScenarioResult "s7_dashscope_err" $true "执行失败且原生错误信息传播到 history"
        } catch {
            Set-ScenarioResult "s7_dashscope_err" $false $_.Exception.Message
        }
    }
    finally {
        # ---- TEARDOWN: 无论成功/异常都执行 ----
        Write-Log "TEARDOWN: 开始清理"
        if ($script:ComfyPID) { Stop-Process -Id $script:ComfyPID -Force -ErrorAction SilentlyContinue }
        if ($script:MockPID)  { Stop-Process -Id $script:MockPID  -Force -ErrorAction SilentlyContinue }
        # 等待两个端口释放 (≤10s)
        foreach ($p in @($MockPort, $ComfyPort)) {
            if (-not (Wait-PortFree -Port $p -TimeoutSec 10)) {
                Write-Log "  WARNING: 端口 $p 10s 后仍被监听"
            }
        }
        # 仅删除本次创建的 junction: rmdir 不带 /S, 绝不对 junction 使用 Remove-Item -Recurse
        if ($script:JunctionCreated) {
            cmd /c "rmdir `"$JunctionPath`""
            if (Test-Path -LiteralPath $JunctionPath) {
                Write-Log "  WARNING: junction 删除后仍存在: $JunctionPath"
            } else {
                Write-Log "  junction 已删除"
            }
            # 完整性 sanity: rmdir 只解开链接, 插件源码必须毫发无损
            if (Test-Path -LiteralPath (Join-Path $PluginDir "urls.py")) {
                Write-Log "  sanity OK: PluginDir\urls.py 仍在 (插件源码未受损)"
            } else {
                # T10 完成前 urls.py 可能合法缺失 → 仅 WARNING, 不判失败
                Write-Log "  WARNING: PluginDir\urls.py 不存在 (T10 之前属正常)"
            }
        } else {
            Write-Log "  junction 非本次创建, 保留原位"
        }
        # 删除本次缓存目录 (普通临时目录, 非 junction, -Recurse 安全)
        Remove-Item -LiteralPath $CacheDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Log "  缓存目录已清理; evidence 目录完整保留: $EvidenceDir"
    }
} catch {
    # STEP 0/1/2/3 的硬中断 (如 mock 不就绪) 传播至此; teardown 已在 finally 中执行
    Write-Log "流程中断: $($_.Exception.Message)"
    $script:FailureNotes.Add("ABORT : $($_.Exception.Message)")
}

# ---- STEP 8: 汇总 ----
$summary = [ordered]@{}
foreach ($k in $script:Results.Keys) { $summary[$k] = [bool]$script:Results[$k] }
$summary["timestamp"] = (Get-Date).ToString("o")
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $EvidenceDir "s0_summary.json")

$failedNames = @($script:Results.Keys | Where-Object { -not $script:Results[$_] })
if ($failedNames.Count -eq 0) {
    Write-Host "SMOKE_PASS"
    exit 0
} else {
    Write-Host ("SMOKE_FAIL: " + ($failedNames -join ", "))
    foreach ($note in $script:FailureNotes) { Write-Host "  - $note" }
    exit 1
}
