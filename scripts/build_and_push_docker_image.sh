TAG="$1"
USERNAME='mmaaz60'
#!/bin/sh

docker build -t $USERNAME/ssl_for_fgvc:"$TAG" .
docker push $USERNAME/ssl_for_fgvc:"$TAG"
