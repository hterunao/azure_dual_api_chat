import uuid
import azure.cosmos.documents as documents
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime, date, timezone, timedelta

# ----------------------------------------------------------------------------------------------------------
# Prerequistes -
#
# 1. An Azure Cosmos account -
#    https://docs.microsoft.com/azure/cosmos-db/create-cosmosdb-resources-portal#create-an-azure-cosmos-db-account
#
# 2. Microsoft Azure Cosmos PyPi package -
#    https://pypi.python.org/pypi/azure-cosmos/
# ----------------------------------------------------------------------------------------------------------
# Sample - demonstrates the basic CRUD operations on a Item resource for Azure Cosmos
# ----------------------------------------------------------------------------------------------------------

class CosmosDB:
    def __init__(self, host, key, database_id, container_id):
        self.client = cosmos_client.CosmosClient(host, {'masterKey': key}, user_agent="CosmosDBPythonQuickstart")
        # setup database for this sample
        try:
            self.db = self.client.create_database(id=database_id)
            print('Database with id \'{0}\' created'.format(database_id))

        except exceptions.CosmosResourceExistsError:
            self.db = self.client.get_database_client(database_id)
            print('Database with id \'{0}\' was found'.format(database_id))

        # setup container for this sample
        try:
            self.container = self.db.create_container(
                id=container_id,
                partition_key=PartitionKey(path='/partitionKey'),
                offer_throughput=400
            )
            print('Container with id \'{0}\' created'.format(container_id))

        except exceptions.CosmosResourceExistsError:
            self.container = self.db.get_container_client(container_id)
            print('Container with id \'{0}\' was found'.format(container_id))

    def log(self, obj, partition_key):
        JST = timezone(timedelta(hours=9))
        iso_datetime = datetime.now(JST).isoformat()
        item = {
            "id": uuid.uuid4().hex,
            "type": "log",
            "partitionKey": partition_key,
            "timestamp": iso_datetime,
            "date": iso_datetime[:10],
            "month": iso_datetime[:7],
            "data": obj
        }
        self.container.create_item(body=item)

