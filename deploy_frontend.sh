echo "Deploying frontend"

echo "Changing directory to frontend"
cd "$(pwd)/backend"

echo "Installing npm dependencies"
npm i

echo "Building the frontend"
npm run build

echo "Copying build files to /var/www/html/"
echo $PASSWORD | sudo -S cp -r dist/* /var/www/html/

echo "Restarting nginx"
echo $PASSWORD | sudo -S systemctl restart nginx

echo "Done deploying frontend"