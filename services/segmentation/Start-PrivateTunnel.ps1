param(
  [Parameter(Mandatory=$true)][string]$HostAddress,
  [Parameter(Mandatory=$true)][string]$User,
  [Parameter(Mandatory=$true)][string]$IdentityFile,
  [Parameter(Mandatory=$true)][string]$HostKeyAlias,
  [Parameter(Mandatory=$true)][string]$TokenFile,
  [string]$BindAddress,
  [int]$Port=18843
)
$ErrorActionPreference='Stop'
if (!(Test-Path -LiteralPath $TokenFile)) {throw 'Private SAM token missing'}
$sshArgs=@('-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-o','BatchMode=yes',
 '-o','StrictHostKeyChecking=yes','-o',('HostKeyAlias='+$HostKeyAlias),'-o','IdentitiesOnly=yes','-i',$IdentityFile,
 '-L',('127.0.0.1:'+ $Port + ':127.0.0.1:'+$Port))
if ($BindAddress) {$sshArgs+=@('-b',$BindAddress)}
$sshArgs+=($User+'@'+$HostAddress)
$process=Start-Process -FilePath 'ssh.exe' -ArgumentList $sshArgs -WindowStyle Hidden -PassThru
$env:IMAGE2PPT_SAM_URL='http://127.0.0.1:'+$Port
$env:IMAGE2PPT_SAM_TOKEN_FILE=[IO.Path]::GetFullPath($TokenFile)
[pscustomobject]@{pid=$process.Id;url=$env:IMAGE2PPT_SAM_URL;token_file=$env:IMAGE2PPT_SAM_TOKEN_FILE} | ConvertTo-Json
