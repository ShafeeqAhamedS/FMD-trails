echo "Deploying FastAPI service"

echo "Changing directory to backend"
cd "$(pwd)/backend"

echo "Moving fastapi.service to /etc/systemd/system/"
sudo cp fastapi.service /etc/systemd/system/ -S $PASSWORD

echo "Reloading systemd manager configuration"
sudo systemctl daemon-reload -S $PASSWORD

echo "Enabling fastapi service"
sudo systemctl enable fastapi -S $PASSWORD

echo "Starting fastapi service"
sudo systemctl start fastapi -S $PASSWORD

sleep 5

echo "Checking status of fastapi service"
sudo systemctl status fastapi -S $PASSWORD

echo "Done deploying FastAPI service"