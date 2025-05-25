import os
import socket
import sys
sys.path.append("/apache/spark/python/lib/py4j-0.10.7-src.zip")
sys.path.append("/apache/spark/python")
os.environ['SPARK_HOME'] = "/data/shpx/data/mpalei/spark-3.1.1.0.1.0-bin-ebay"
os.environ["PYSPARK_PYTHON"] = "/usr/share/anaconda3/python3.7/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/share/anaconda3/python3.7/bin/python"

from pyspark.sql import SparkSession
import random
from multiprocessing import Process

def spark_session(app_name, queue="hddq-exprce-perso-high-mem"):
    
    spark = (
        SparkSession.builder.appName(app_name)
            .master("yarn")
            .config("spark.driver.maxResultSize", "0")  # unlimited
            .config("spark.driver.host", socket.gethostbyname(socket.gethostname()))
            .config("spark.driver.port", "30202")
            .config("spark.executor.memoryOverhead", "4096")
            .config("spark.executor.cores", "3")
            .config("spark.driver.memory", "20g")  # "9g" --> try to increase?
            .config("spark.executor.memory", "20g")  # "25g"
            .config("spark.rdd.compress", True)
            .config("spark.network.timeout", "600s")
            .config("spark.executor.heartbeatInterval", "300s")
            .config("spark.sql.broadcastTimeout", "2000s")
            .config("spark.dynamicAllocation.minExecutors", 0)
            .config("spark.dynamicAllocation.initialExecutors", 10)  # 10
            .config("spark.dynamicAllocation.maxExecutors", 100)  # 100
            .config("spark.sql.shuffle.partitions", 1000)  # new
            .config("spark.yarn.queue", queue)
            .config("spark.speculation", False)
            .config("spark.hadoop.mapreduce.map.speculative", False)
            .config("spark.hadoop.mapreduce.reduce.speculative", False)
            .config("spark.kryoserializer.buffer.max", "1g")
            .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
            .config("spark.storage.memoryFraction", 0)
           
            .config("spark.files", "viewfs://apollo-rno/user/b_zeta_devsuite/files/aes.properties") 
            .config("spark.files", "/apache/hive/conf/hive-site.xml,viewfs://apollo-rno/user/b_zeta_devsuite/files/aes.properties") 
            .config("spark.files", "/data/shpx/notebooks/hroitman/Purchase-Suppression/feature_helpers.py") 
        
            .config("spark.yarn.queue", "hddq-exprce-perso-high-mem")
            .config("spark.ui.prometheus.enabled", True)
            .config("spark.files", "/apache/hive/conf/hive-site.xml")
            .config("spark.jars", "viewfs://apollo-rno/apps/b_hive_dba/spark-binary/iceberg-spark-runtime-3.1_2.12-1.2.1.0.7.1.jar")
        
         # Adding Iceberg configurations
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            #.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.connector.catalog.DelegatingCatalog") 
            .config("spark.sql.catalog.spark_catalog.delegating.provider.iceberg", "DSv2HmsCatalog")
            .config("spark.sql.catalog.DSv2HmsCatalog", "org.apache.iceberg.spark.DSv2HmsCatalog")
            
            .config("spark.files", "viewfs://apollo-rno/user/b_zeta_devsuite/files/aes.properties") 
            .config("spark.files", "/apache/hive/conf/hive-site.xml,viewfs://apollo-rno/user/b_zeta_devsuite/files/aes.properties") 
            .config("spark.files", "/data/shpx/notebooks/hroitman/Purchase-Suppression/feature_helpers.py") 
        
            
        
            .enableHiveSupport()
            .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("checkpoint/")
    return spark


def save_table(session, df, table_name):
    df.createOrReplaceTempView(f"{table_name}_tmp") 
    session.sql(f"drop table if exists {table_name}")
    session.sql(f"create table {table_name} as select * from {table_name}_tmp")


##########################################
    
import os
import socket
import sys
sys.path.append("/apache/spark/python/lib/py4j-0.10.7-src.zip")
sys.path.append("/apache/spark/python")
os.environ['SPARK_HOME'] = "/data/shpx/data/mpalei/spark-3.1.1.0.1.0-bin-ebay"
os.environ["PYSPARK_PYTHON"] = "/usr/share/anaconda3/python3.7/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/share/anaconda3/python3.7/bin/python"
from pyspark.sql import SparkSession
import random
from multiprocessing import Process

def spark_session(app_name, queue="hddq-exprce-perso-high-mem"):
    
    spark = (
        SparkSession.builder.appName(app_name)
            .master("yarn")
            .config("spark.driver.maxResultSize", "0") # unlimited
            .config("spark.driver.host", socket.gethostbyname(socket.gethostname()))
            .config("spark.driver.port", "30202")
            .config("spark.executor.memoryOverhead", "4096")
            .config("spark.executor.cores", "3")
            .config("spark.driver.memory", "20g") #"9g" --> try to increae?
            .config("spark.executor.memory", "20g") #"25g"
            .config("spark.rdd.compress", True)
            .config("spark.network.timeout", "600s")
            .config("spark.executor.heartbeatInterval", "300s")
            .config("spark.sql.broadcastTimeout", "2000s")
            # disable autoBroadcastJoin
            #.config("spark.sql.autoBroadcastJoinThreshold", -1) 
            .config("spark.dynamicAllocation.minExecutors", 0)
            .config("spark.dynamicAllocation.initialExecutors", 10) #10
            .config("spark.dynamicAllocation.maxExecutors", 100) #100
            .config("spark.sql.shuffle.partitions", 1000) #new
            .config("spark.yarn.queue", queue)
            ## speculative execution off
            .config("spark.speculation", False)
            .config("spark.hadoop.mapreduce.map.speculative", False)
            .config("spark.hadoop.mapreduce.reduce.speculative", False)
            .config("spark.kryoserializer.buffer.max", "1g")
            .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
            ## not using caching - so don't save memory for cahce
            .config("spark.storage.memoryFraction", 0)
            .config("spark.files", "viewfs://apollo-rno/user/b_zeta_devsuite/files/aes.properties") 
            .config("spark.files", "/apache/hive/conf/hive-site.xml,viewfs://apollo-rno/user/b_zeta_devsuite/files/aes.properties") 
            .config("spark.files", "/data/shpx/notebooks/hroitman/Purchase-Suppression/feature_helpers.py") 
            .enableHiveSupport()
            .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("checkpoint/")
    return spark


def save_table(session, df, table_name):
    df.createOrReplaceTempView(f"{table_name}_tmp") 
    session.sql(f"drop table if exists {table_name}")
    session.sql(f"create table {table_name} as select * from {table_name}_tmp")


