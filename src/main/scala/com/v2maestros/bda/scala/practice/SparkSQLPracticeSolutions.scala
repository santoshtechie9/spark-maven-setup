package com.v2maestros.bda.scala.practice
/*
-----------------------------------------------------------------------------

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Practice Solutions : Spark with Scala - SQL
-----------------------------------------------------------------------------
*/
object SparkSQLPracticeSolutions extends App{
	
  	import com.v2maestros.bda.scala.common._
	import org.apache.spark.sql.functions._
	import org.apache.log4j.Logger
	import org.apache.log4j.Level;
	
	Logger.getLogger("org").setLevel(Level.ERROR);
	Logger.getLogger("akka").setLevel(Level.ERROR);
		
  	val spSession = SparkCommonUtils.spSession
  	val spContext = SparkCommonUtils.spContext
  	val datadir = SparkCommonUtils.datadir
  	
	/*----------------------------------------------------------------------
	  		# Spark Data Frames
	 ---------------------------------------------------------------------*/
		
	// 1. Your dataset has a file iris.csv. load it into a data frame irisDF
	//	All Column Datatypes other than for Species should be of Double type
	// Print the contents and the schema.
		
	//Create the schema for the data to be loaded into Dataset.  
  	//Load the CSV file into a RDD
  	println("Loading data file :")
	val irisData = spContext.textFile(datadir + "iris.csv")
	irisData.cache()
	irisData.take(5)
	
	//Remove the first line (contains headers)
	val dataLines = irisData.filter(x =>  !x.contains("Sepal"))
	dataLines.count()
	
	import org.apache.spark.ml.linalg.{Vector, Vectors}
	import org.apache.spark.ml.feature.LabeledPoint
	import org.apache.spark.sql.Row;
	import org.apache.spark.sql.types._
  		//Schema for Data Frame
	val schema =
	  StructType(
	    StructField("SPECIES", StringType, false) ::
	    StructField("SEPAL_LENGTH", DoubleType, false) ::
	    StructField("SEPAL_WIDTH", DoubleType, false) ::
	    StructField("PETAL_LENGTH", DoubleType, false) ::
	    StructField("PETAL_WIDTH", DoubleType, false) :: Nil)
	
	def transformToNumeric( inputStr : String) : Row = {
	    val attList=inputStr.split(",")

	    //Filter out columns not wanted at this stage
	    val values= Row(attList(4), 
	                     attList(0).toDouble,  
	                     attList(1).toDouble,  
	                     attList(2).toDouble,  
	                     attList(3).toDouble 
	                     )
	    return values
	 }   
	
	//Change to a Vector
	val irisVectors = dataLines.map(transformToNumeric)
	irisVectors.collect()

	println("Transformed data in Data Frame")
    val irisDF = spSession.createDataFrame(irisVectors, schema)
    irisDF.printSchema()
    irisDF.show(5)
    
    /*-----------------------------------------------------------------------------
	2. In the irisDF, filter for rows whose PetalWidth is greater than 0.4
	and count them.
	-----------------------------------------------------------------------------*/
	println("Petal width > 4 = " + irisDF.filter(irisDF("PETAL_WIDTH") > 0.4).count())
	
	/*-----------------------------------------------------------------------------
	3. Register a temp table called "iris" using irisDF. Then find average
	Petal Width by Species using that table.
	-----------------------------------------------------------------------------*/
	irisDF.createOrReplaceTempView("iris")
	spSession.sql("select SPECIES,avg(PETAL_WIDTH) from iris group by SPECIES").show()
	
	/*-----------------------------------------------------------------------------
	Hope you had some good practice !! Recommend trying out your own use cases
	-----------------------------------------------------------------------------*/

}