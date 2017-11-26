package com.v2maestros.bda.scala.train

object YourFirstSparkProgram extends App{
  
	import org.apache.spark.SparkContext
	import org.apache.spark.SparkConf
	
	//A name for the spark instance. Can be any string
	val appName="V2Maestros"
	//Pointer / URL to the Spark instance - embedded
	//val sparkMasterURL = "local[2]"
	val sparkMasterURL = "spark://169.254.116.226:7077"
	
	
	//Create a configuration object
	val conf = new SparkConf()
			.setAppName(appName)
			.setMaster(sparkMasterURL)
			.set("spark.executor.memory","2g")
			
	//Start a Spark Session
	val spContext = SparkContext.getOrCreate(conf)
	
	//Check http://localhost:4040
	
	//Load a data file into an RDD
	val tweetsRDD = spContext.textFile("C:/Users/kumaran/Dropbox/" +
			"V2Maestros/ScalaWorkSpace/scala-spark-bda" +
			"/data-files/movietweets.csv")

	//print first five lines
	for( tweet <- tweetsRDD.take(5)) println(tweet)
	
	//Print number of lines in file
	//This is lazy evaluation
	println("Total tweets in file :" + tweetsRDD.count())
	
	//Convert tweets to upper case
	val tweetsUpper = tweetsRDD.map( s => s.toUpperCase())
	
	//Print the converted items.
	tweetsUpper.take(5)

}