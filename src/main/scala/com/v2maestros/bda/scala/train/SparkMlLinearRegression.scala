/*
-----------------------------------------------------------------------------

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Machine Learning - Linear Regression

Problem Statement
*****************
The input data set contains data about details of various car 
models. Based on the information provided, the goal is to come up 
with a model to predict Miles-per-gallon of a given model.

Techniques Used:

1. Linear Regression ( multi-variate)
2. Data Imputation - replacing non-numeric data with numeric ones
3. Variable Reduction - picking up only relevant features

-----------------------------------------------------------------------------
*/

package com.v2maestros.bda.scala.train

object SparkMlLinearRegression extends App {
	
	import com.v2maestros.bda.scala.common._
	import org.apache.spark.sql.functions._
	import org.apache.log4j.Logger
	import org.apache.log4j.Level;
	
	//Logger setup to avoid info log flood
	Logger.getLogger("org").setLevel(Level.ERROR);
	Logger.getLogger("akka").setLevel(Level.ERROR);
		
  	val spSession = SparkCommonUtils.spSession
  	val spContext = SparkCommonUtils.spContext
  	val datadir = SparkCommonUtils.datadir
  	
  	//Load the CSV file into a RDD
	val autoData = spContext.textFile(datadir 
					+ "auto-miles-per-gallon.csv")
	autoData.cache()
	
	//Remove the first line (contains headers)
	val dataLines = autoData.filter(x => 
						!x.contains( "CYLINDERS"))
	println("\nTotal lines in data : " + dataLines.count())
	
	//Convert the RDD into a Dense Vector. As a part of this exercise
	//   1. Remove unwanted columns
	//   2. Change non-numeric ( values=? ) to numeric
	//Use default for average HP
	
	import org.apache.spark.ml.linalg.{Vector, Vectors}
	import org.apache.spark.ml.feature.LabeledPoint
	import org.apache.spark.sql.Row;
	import org.apache.spark.sql.types._
	
	//Schema for data frame
	val schema =
	  StructType(
	    StructField("MPG", DoubleType, false) ::
	    StructField("CYLINDERS", DoubleType, false) ::
	    StructField("HP", DoubleType, false) ::
	    StructField("ACCELERATION", DoubleType, false) ::
	    StructField("MODELYEAR", DoubleType, false) :: Nil)
	
	val avgHP =spContext.broadcast(80.0)
	
	//function to transform data to numeric values
	def transformToNumeric( inputStr : String) : Row = {
	
	    val attList=inputStr.split(",") 
	    //Replace ? values with a normal value
	    var hpValue = attList(3)
	    if (hpValue.contains("?")) {
	        hpValue=avgHP.value.toString
	    }
	    //Filter out columns not wanted at this stage
	    val values = Row((attList(0).toDouble ), 
	                     attList(1).toDouble,  
	                     hpValue.toDouble,    
	                     attList(5).toDouble,  
	                     attList(6).toDouble
	                     )
	    return values
	}
	//Keep only MPG, CYLINDERS, HP,ACCELERATION and MODELYEAR
	val autoVectors = dataLines.map(transformToNumeric)
	
	println("\nCleansed and Transformed data : " )
	autoVectors.collect()
	
	//Convert to Data Frame
	val autoDf = spSession
			.createDataFrame(autoVectors, schema)
	autoDf.show(5)
	
	//Perform Correlation Analysis
	println("\nCorrelation Analysis : " )
	for ( field <- schema.fields ) {
		if ( ! field.dataType.equals(StringType)) {
			println("Correlation between MPG and " 
						+ field.name +
					 " = " + autoDf.stat.corr(
							 "MPG", field.name))
		}
	}
	
	//Transform to a Data Frame for input to Machine Learing
	//Drop columns that are not required (low correlation)
	def transformToLabelVectors(inStr : Row ) : LabeledPoint = { 
	    val labelVectors = new LabeledPoint(
	    					inStr.getDouble(0) , 
							Vectors.dense(inStr.getDouble(1),
									inStr.getDouble(2),
									inStr.getDouble(3),
									inStr.getDouble(4)));
	    return labelVectors
	}
	val tempRdd1 = autoDf.rdd.repartition(2);
	val autoLabelVectors 
			= tempRdd1.map(transformToLabelVectors)
	autoLabelVectors.collect()
	    
	val autoDF = spSession
		.createDataFrame(autoLabelVectors, 
				classOf[LabeledPoint] )
	
	println("\nML Ready data : " )
	autoDF.select("label","features").show(10)
	
	//Split into training and testing data
	val Array(trainingData, testData) 
		= autoDF.randomSplit(Array(0.9, 0.1))
	println("\nTraining Data count : " + trainingData.count())
	println("\nTest Data count : " + testData.count())
	
	//Build the model on training data
	import org.apache.spark.ml.regression.LinearRegression
	
	val lr = new LinearRegression().setMaxIter(10)
	val lrModel = lr.fit(trainingData)
	
	println("\nCoefficients: " + lrModel.coefficients)
	print("\nIntercept: " + lrModel.intercept)
	lrModel.summary.r2
	
	//Predict on the test data
	val predictions = lrModel.transform(testData)
	println("\nPredictions : " )
	predictions.select("prediction","label","features").show()
	
	//Evaluate the results. Find R2
	import org.apache.spark.ml.evaluation.RegressionEvaluator
	val evaluator = new RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("label")
	evaluator.setMetricName("r2")
	println("\nAccuracy = " + evaluator.evaluate(predictions))

  
}