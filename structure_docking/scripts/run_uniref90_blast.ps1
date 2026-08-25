param(
    [Parameter(Mandatory = $true)][string]$Email,
    [string]$InputJson = "structure_docking/01_rank_pool\rank_1_200_for_uniprot.json",
    [string]$OutputDir = "structure_docking/02_uniref90_blast",
    [int]$StartRank = 1,
    [int]$EndRank = 200,
    [int]$BatchSize = 30,
    [int]$PollSeconds = 15
)

$ErrorActionPreference = "Stop"
$baseUrl = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"
$data = Get-Content -LiteralPath $InputJson -Raw | ConvertFrom-Json
$records = @($data.records | Where-Object { $_.rank -ge $StartRank -and $_.rank -le $EndRank })
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$statePath = Join-Path $OutputDir "jobs.json"
$jobs = @{}

if (Test-Path -LiteralPath $statePath) {
    $saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    foreach ($job in @($saved.jobs)) {
        $jobs[[int]$job.rank] = $job
    }
}

function Save-State {
    $payload = [ordered]@{
        service = "EMBL-EBI NCBI BLAST REST"
        database = "uniref90"
        start_rank = $StartRank
        end_rank = $EndRank
        updated_at = (Get-Date).ToString("o")
        jobs = @($jobs.Values | Sort-Object { [int]$_.rank })
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Update-JobStatus([object]$job) {
    try {
        $status = (Invoke-RestMethod -Uri "$baseUrl/status/$($job.job_id)" -TimeoutSec 45).Trim()
        $job.status = $status
        $job.checked_at = (Get-Date).ToString("o")
        if ($status -eq "FINISHED" -and -not $job.result_file) {
            $rankText = ([int]$job.rank).ToString("0000")
            $xmlPath = Join-Path $OutputDir "rank_${rankText}.xml"
            try {
                Invoke-WebRequest -Uri "$baseUrl/result/$($job.job_id)/xml" -UseBasicParsing -OutFile $xmlPath -TimeoutSec 90
                $job.result_file = $xmlPath
                $job.result_type = "xml"
            } catch {
                $outPath = Join-Path $OutputDir "rank_${rankText}.out.txt"
                Invoke-WebRequest -Uri "$baseUrl/result/$($job.job_id)/out" -UseBasicParsing -OutFile $outPath -TimeoutSec 90
                $job.result_file = $outPath
                $job.result_type = "out"
            }
        }
    } catch {
        $job.last_error = $_.Exception.Message
        $job.checked_at = (Get-Date).ToString("o")
    }
}

while ($true) {
    $active = @($jobs.Values | Where-Object { $_.status -notin @("FINISHED", "ERROR", "FAILURE", "NOT_FOUND") })
    foreach ($job in $active) {
        Update-JobStatus $job
    }
    Save-State

    $active = @($jobs.Values | Where-Object { $_.status -notin @("FINISHED", "ERROR", "FAILURE", "NOT_FOUND") })
    $notSubmitted = @($records | Where-Object { -not $jobs.ContainsKey([int]$_.rank) })

    $freeSlots = [Math]::Max(0, $BatchSize - $active.Count)
    if ($freeSlots -gt 0 -and $notSubmitted.Count -gt 0) {
        $toSubmit = @($notSubmitted | Select-Object -First $freeSlots)
        foreach ($record in $toSubmit) {
            $rank = [int]$record.rank
            $body = @{
                email = $Email
                title = "MDS_rank_$($rank.ToString('0000'))"
                program = "blastp"
                database = "uniref90"
                sequence = [string]$record.protein_sequence
                stype = "protein"
                alignments = "50"
                scores = "50"
                exp = "1e-5"
                matrix = "BLOSUM62"
                filter = "F"
            }
            try {
                $jobId = (Invoke-RestMethod -Uri "$baseUrl/run" -Method Post -Body $body -TimeoutSec 60).Trim()
                $jobs[$rank] = [pscustomobject]@{
                    rank = $rank
                    score = [double]$record.y_pred
                    job_id = $jobId
                    status = "SUBMITTED"
                    submitted_at = (Get-Date).ToString("o")
                    checked_at = $null
                    result_file = $null
                    result_type = $null
                    last_error = $null
                }
                Write-Output "Submitted rank $rank as $jobId"
            } catch {
                Write-Output "Submission failed for rank ${rank}: $($_.Exception.Message)"
                Start-Sleep -Seconds 5
                break
            }
            Save-State
            Start-Sleep -Seconds 1
        }
        continue
    }

    $finished = @($jobs.Values | Where-Object { $_.status -eq "FINISHED" -and $_.result_file }).Count
    $failed = @($jobs.Values | Where-Object { $_.status -in @("ERROR", "FAILURE", "NOT_FOUND") }).Count
    Write-Output "Progress: submitted=$($jobs.Count), active=$($active.Count), downloaded=$finished, failed=$failed, remaining=$($notSubmitted.Count)"

    if ($active.Count -eq 0 -and $notSubmitted.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Save-State
Write-Output "UniRef90 BLAST workflow finished. State: $statePath"
