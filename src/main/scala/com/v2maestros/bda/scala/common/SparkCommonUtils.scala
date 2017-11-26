/****************************************************************************

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Common library used by other programs
*****************************************************************************/

package com.v2maestros.bda.scala.common

object SparkCommonUtils {
  
	import org.apache.spark.sql.SparkSession
	import org.apache.spark.SparkContext
	import org.apache.spark.SparkConf
	
	//Directory where the data files for the examples exist.
	val datadir = "C:/Users/kumaran/Dropbox/V2Maestros/ScalaWorkSpace/scala-spark-bda/data-files/"

	//A name for the spark instance. Can be any string
	val appName="V2Maestros"
	//Pointer / URL to the Spark instance - embedded
	val sparkMasterURL = "local[2]"
	//Temp dir required for Spark SQL
	val tempDir= "file:///c:/temp/spark-warehouse"
	
	var spSession:SparkSession = null
	var spContext:SparkContext = null
	 
	//Initialization. Runs when object is created
	{	
		//Need to set hadoop.home.dir to avoid errors during startup
		System.setProperty("hadoop.home.dir", 
				"c:\\spark\\winutils\\");
		
		//Create spark configuration object
		val conf = new SparkConf()
			.setAppName(appName)
			.setMaster(sparkMasterURL)
			.set("spark.executor.memory","2g")
			.set("spark.sql.shuffle.partitions","2")
	
		//Get or create a spark context. Creates a new instance if not exists	
		spContext = SparkContext.getOrCreate(conf)

		//Create a spark SQL session
		spSession = SparkSession
		  .builder()
		  .appName(appName)
		  .master(sparkMasterURL)
		  .config("spark.sql.warehouse.dir", tempDir)  
		  .getOrCreate()

	}
}