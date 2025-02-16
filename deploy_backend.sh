echo "Deploying FastAPI service"

echo "Changing directory to backend"
cd "$(pwd)/backend"

echo "Moving fastapi.service to /etc/systemd/system/"
sudo cp fastapi.service /etc/systemd/system/

echo "Reloading systemd manager configuration"
sudo systemctl daemon-reload

echo "Enabling fastapi service"
sudo systemctl enable fastapi

echo "Starting fastapi service"
sudo systemctl start fastapi

sleep 5

echo "Checking status of fastapi service"
sudo systemctl status fastapi

echo "Done deploying FastAPI service"