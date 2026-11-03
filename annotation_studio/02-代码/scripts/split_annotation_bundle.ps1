param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Destination,
    [long]$PartBytes = 200000000
)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
if ($PartBytes -le 0) { throw 'PartBytes must be positive.' }
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$outputDir = (Resolve-Path -LiteralPath $Destination).Path
if (Get-ChildItem -LiteralPath $outputDir -Filter 'AIE3905.zip.*') {
    throw 'Parts already exist. Choose a new output directory.'
}
$originalHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceStream = [System.IO.File]::OpenRead($sourcePath)
$copyBuffer = New-Object byte[] (4 * 1024 * 1024)
$partNames = @()
$checksumLines = @()
try {
    $partIndex = 0
    while ($sourceStream.Position -lt $sourceStream.Length) {
        $partIndex++
        $partName = 'AIE3905.zip.{0:D3}' -f $partIndex
        $partPath = Join-Path $outputDir $partName
        $partStream = [System.IO.File]::Open($partPath, [System.IO.FileMode]::CreateNew)
        $remaining = $PartBytes
        try {
            while ($remaining -gt 0) {
                $count = $sourceStream.Read($copyBuffer, 0, [int][Math]::Min($copyBuffer.Length, $remaining))
                if ($count -eq 0) { break }
                $partStream.Write($copyBuffer, 0, $count)
                $remaining -= $count
            }
        } finally { $partStream.Dispose() }
        $partHash = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $partNames += $partName
        $checksumLines += "$partHash  $partName"
        Write-Host "$partName - $((Get-Item -LiteralPath $partPath).Length) bytes"
    }
} finally { $sourceStream.Dispose() }
# Read the actual parts back in order, without creating another 1.1 GB merged copy.
$verifyHash = [System.Security.Cryptography.SHA256]::Create()
try {
    foreach ($partName in $partNames) {
        $partStream = [System.IO.File]::OpenRead((Join-Path $outputDir $partName))
        try {
            while (($count = $partStream.Read($copyBuffer, 0, $copyBuffer.Length)) -gt 0) {
                [void]$verifyHash.TransformBlock($copyBuffer, 0, $count, $copyBuffer, 0)
            }
        } finally { $partStream.Dispose() }
    }
    [void]$verifyHash.TransformFinalBlock([byte[]]@(), 0, 0)
    $joinedHash = ([BitConverter]::ToString($verifyHash.Hash)).Replace('-', '').ToLowerInvariant()
    if ($joinedHash -ne $originalHash) { throw 'Split verification failed.' }
} finally { $verifyHash.Dispose() }
[IO.File]::WriteAllText((Join-Path $outputDir 'SHA256SUMS'), (($checksumLines -join "`n") + "`n"), [Text.Encoding]::ASCII)
[IO.File]::WriteAllText((Join-Path $outputDir 'package.sha256'), "$originalHash  AIE3905.zip`n", [Text.Encoding]::ASCII)
Write-Host "Verified: $($partNames.Count) parts match the original ZIP exactly."
