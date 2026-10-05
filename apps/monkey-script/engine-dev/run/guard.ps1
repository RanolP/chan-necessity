(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm';StartTime=(Get-Date).AddMinutes(-5)} -ErrorAction SilentlyContinue).Count
Get-CimInstance Win32_Process -Filter "name='firefox.exe'" | Where-Object { $_.CommandLine -match 'cb-v044|cb-shaderopt' } | Measure-Object | ForEach-Object { "other-ff:" + $_.Count }
