$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    "username" = "testuser"
    "password" = "Test@1234"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/register" -Method POST -Headers $headers -Body $body