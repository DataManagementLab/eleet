docker run -p 27117:27017 -v ~/.mongodb-data:/data/db -d --name murban-mongo     -e MONGO_INITDB_ROOT_USERNAME=murban     -e MONGO_INITDB_ROOT_PASSWORD=FfoORxeeYwl5xeH6ziMh     --restart always mongo
docker exec -i  murban-mongo sh -c "exec mongorestore --authenticationDatabase admin -u murban -p FfoORxeeYwl5xeH6ziMh --archive --nsInclude wikidata.wiki_graph --gzip" < wikidata.dump
