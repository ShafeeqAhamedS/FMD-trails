echo "Deploying FastAPI service"

echo "Changing directory to backend"
cd "$(pwd)/backend"

echo "Moving fastapi.service to /etc/systemd/system/"
echo $PASSWORD | sudo -S cp fastapi.service /etc/systemd/system/

echo "Reloading systemd manager configuration"
echo $PASSWORD | sudo -S systemctl daemon-reload

echo "Enabling fastapi service"
echo $PASSWORD | sudo -S systemctl enable fastapi

echo "Starting fastapi service"
echo $PASSWORD | sudo -S systemctl start fastapi

sleep 5

echo "Checking status of fastapi service"
echo $PASSWORD | sudo -S systemctl status fastapi

echo "Done deploying FastAPI service"
