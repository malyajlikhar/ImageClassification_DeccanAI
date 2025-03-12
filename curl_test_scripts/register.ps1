$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    "username" = "testuser"
    "password" = "Test@1234"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8001/register" -Method POST -Headers $headers -Body $body