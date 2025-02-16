echo "Deploying frontend"

echo "Changing directory to frontend"
cd "$(pwd)/backend"

echo "Installing npm dependencies"
npm i

echo "Building the frontend"
npm run build

echo "Copying build files to /var/www/html/"
sudo cp -r dist/* /var/www/html/

echo "Restarting nginx"
sudo systemctl restart nginx

echo "Done deploying frontend"