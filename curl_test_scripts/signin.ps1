$credPair = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("testuser:Test@1234"))
$headers = @{
    "Authorization" = "Basic $credPair"
}

Invoke-WebRequest -Uri "http://localhost:8000/signin" -Method POST -Headers $headers