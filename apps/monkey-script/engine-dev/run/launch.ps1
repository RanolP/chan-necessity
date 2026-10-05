$f=Join-Path $env:TEMP 'cb-engine-ff'
New-Item -ItemType Directory -Force -Path $f | Out-Null
Set-Content -Path "$f\user.js" -Value @('user_pref("media.volume_scale", "0.0");','user_pref("browser.shell.checkDefaultBrowser", false);','user_pref("datareporting.policy.dataSubmissionEnabled", false);','user_pref("browser.aboutwelcome.enabled", false);','user_pref("privacy.reduceTimerPrecision", false);')
$p = Start-Process 'C:\Program Files\Firefox Developer Edition\firefox.exe' -ArgumentList '-no-remote','-profile',$f,'--remote-debugging-port','9531' -PassThru
$p.Id
