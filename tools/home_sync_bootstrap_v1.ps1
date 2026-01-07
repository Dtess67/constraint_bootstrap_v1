Write-Host "--- Environment Info ---"
python --version
git --version

Write-Host "`n--- Git Status ---"
git status

$remote = git remote
if (-not $remote) {
    Write-Warning "No remote exists. Stopping sync."
    exit
}

Write-Host "`n--- Pulling with Rebase ---"
git pull --rebase origin master

if (-not (git diff --quiet) -or -not (git diff --cached --quiet)) {
    Write-Host "`n--- Committing changes ---"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git add .
    git commit -m "Home sync: $timestamp"
}

Write-Host "`n--- Pushing to Remote ---"
$currentBranch = git branch --show-current
git push origin $currentBranch
