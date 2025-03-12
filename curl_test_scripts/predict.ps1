$credPair = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("testuser:Test@1234"))
curl.exe -X POST "http://localhost:8001/predict" -H "Authorization: Basic $credPair" -F "file=@C:\Projects\ImageClassification_DeccanAI\test_images\test1.png"
