from pymongo import MongoClient
from pymongo.server_api import ServerApi

# ❗ Replace <db_password> with your real password or use an environment variable
uri = "mongodb+srv://rajusubba14_db_user:Admin123@cluster0.6ypodjn.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

try:
    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
finally:
    client.close()