package com.v2maestros.bda.scala.train
/*
  				 Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Machine Learning - Recommendation Engine

Problem Statement
*****************
The input data contains a file with user, item and ratings. 
The purpose of the exercise is to build a recommendation model
and then predict the affinity for users to various items
-----------------------------------------------------------------------------
*/
object SparkMlRecommendations extends App {
	
	import com.v2maestros.bda.scala.common._
	import org.apache.spark.sql.functions._
	import org.apache.log4j.Logger
	import org.apache.log4j.Level;
	
	Logger.getLogger("org").setLevel(Level.ERROR);
	Logger.getLogger("akka").setLevel(Level.ERROR);
		
  	val spSession = SparkCommonUtils.spSession
  	val spContext = SparkCommonUtils.spContext
  	val datadir = SparkCommonUtils.datadir
  	
  	//Load the CSV file into a RDD
  	println("Loading data file :")
	val ratingsData = spContext.textFile(datadir 
					+ "useritemdata.txt")
	ratingsData.cache()
	ratingsData.take(5)
	
	//Convert the RDD into a Dense Vector. As a part of this exercise
	//   1. Change labels to numeric ones
	
	import org.apache.spark.ml.linalg.{Vector, Vectors}
	import org.apache.spark.ml.feature.LabeledPoint
	import org.apache.spark.sql.Row;
	import org.apache.spark.sql.types._
	
	//Schema for Data Frame
	val schema =
	  StructType(
	    StructField("user", IntegerType, false) ::
	    StructField("item", IntegerType, false) ::
	    StructField("rating", DoubleType, false)  :: Nil)
	
	def transformToNumeric( inputStr : String) : Row = {
	    val attList=inputStr.split(",")

	    //Filter out columns not wanted at this stage
	    val values= Row( attList(0).toInt,  
	                     attList(1).toInt,  
	                     attList(2).toDouble
	                     )
	    return values
	 }   
	
	//Change to a Vector
	val ratingsVectors = ratingsData.map(transformToNumeric)
	ratingsVectors.collect()

	println("Transformed data in Data Frame")
    val ratingsDf = spSession.createDataFrame(ratingsVectors, schema)
    ratingsDf.printSchema()
    ratingsDf.show(5)
    
   	//Split into training and testing data
	val Array(trainingData, testData) 
			= ratingsDf.randomSplit(Array(0.9, 0.1))
	trainingData.count()
	testData.count()
	
	import org.apache.spark.ml.recommendation.ALS
	val als = new ALS()
	als.setRank(10)
	als.setMaxIter(5)
	als.setUserCol("user")
	als.setItemCol("item")
	als.setRatingCol("rating")
	val model = als.fit(trainingData)
	
	val predictions=model.transform(testData)
	println("Recommendation scores:")
	predictions.select("user","item","prediction").show()
  
}