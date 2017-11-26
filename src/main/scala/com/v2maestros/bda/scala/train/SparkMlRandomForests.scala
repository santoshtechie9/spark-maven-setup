package com.v2maestros.bda.scala.train
/*
  				 Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Machine Learning - Random Forests

Problem Statement
*****************
The input data contains surveyed information about potential 
customers for a bank. The goal is to build a model that would 
predict if the prospect would become a customer of a bank, 
if contacted by a marketing exercise.

//// Techniques Used

1. Random Forests
2. Training and Testing
3. Confusion Matrix
4. Indicator Variables
5. Variable Reduction

-----------------------------------------------------------------------------
*/
object SparkMlRandomForests extends App{
  
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
	val bankData = spContext.textFile(datadir + "/bank.csv")
	bankData.cache()
	bankData.count()
	
	//Remove the first line (contains headers)
	val firstLine=bankData.first()
	val dataLines = bankData.filter(x => x != firstLine)
	println("\nTotal lines in data " + 	dataLines.count())
	
		
	import org.apache.spark.ml.linalg.{Vector, Vectors}
	import org.apache.spark.ml.feature.LabeledPoint
	import org.apache.spark.sql.Row;
	import org.apache.spark.sql.types._
	
	//Schema for Data Frame
	val schema =
	  StructType(
	    StructField("OUTCOME", DoubleType, false) ::
	    StructField("AGE", DoubleType, false) ::
	    StructField("SINGLE", DoubleType, false) ::
	    StructField("MARRIED", DoubleType, false) ::
	    StructField("DIVORCED", DoubleType, false) ::
	    StructField("PRIMARY", DoubleType, false) ::
	    StructField("SECONDARY", DoubleType, false) ::
	    StructField("TERTIARY", DoubleType, false) ::
	    StructField("DEFAULT", DoubleType, false) ::
	    StructField("BALANCE", DoubleType, false) ::
	    StructField("LOAN", DoubleType, false) :: Nil)
	    
	def transformToNumeric( inputStr : String) : Row = {
		
	    val attList=inputStr.split(";")
	    
	    val age=attList(0).toDouble
	    //convert outcome to float    
	    val outcome:Double = attList(16).contains("no") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }
	  
	    //create indicator variables for single/married 
	    val single:Double = attList(2).contains("single") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }
	    val married:Double = attList(2).contains("married") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                } 
	    val divorced:Double = attList(2).contains("divorced") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }               
	    
	    //create indicator variables for education
	    val primary:Double = attList(3).contains("primary") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }           
	    val secondary:Double = attList(3).contains("secondary") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }           
	    val tertiary:Double = attList(3).contains("tertiary") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }           
	   
	    //convert default to float
	    val default:Double = attList(4).contains("no") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }
	    //convert balance amount to float
	    val balance:Double = attList(5).contains("no") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }
	    //convert loan to float
	    val loan:Double = attList(7).contains("no") match {
	                    case  true => 1.0
	                    case  false    => 0.0
	                }
	    //Filter out columns not wanted at this stage
	    val values= Row(outcome, age, single, married, 
	                divorced, primary, secondary, tertiary,
	                default, balance, loan )
	    return values
	}
	
	//Change to a Vector
	val bankRDD = dataLines.map(transformToNumeric)
	bankRDD.collect()

	println("\nTransformed data in Data Frame")
    val bankDf = spSession.createDataFrame(bankRDD, schema)
    bankDf.printSchema()
    bankDf.show(5)
    
   	println("\nCorrelation Analysis :")
   	for ( field <- schema.fields ) {
		if ( ! field.dataType.equals(StringType)) {
			println("Correlation between OUTCOME and " + field.name +
					 " = " + bankDf.stat.corr("OUTCOME", field.name))
		}
	}
	
	//Transform to a Data Frame for input to Machine Learing
	//Drop columns that are not required (low correlation / strings)
	def transformToLabelVectors(inStr : Row ) : LabeledPoint = { 
		
	    val labelVectors = new LabeledPoint(inStr.getDouble(0) , 
								Vectors.dense(inStr.getDouble(1),
										inStr.getDouble(2),
										inStr.getDouble(3),
										inStr.getDouble(4),
										inStr.getDouble(5),
										inStr.getDouble(6),
										inStr.getDouble(7),
										inStr.getDouble(8),
										inStr.getDouble(9),
										inStr.getDouble(10)));
	    return labelVectors
	}
	val tempRdd1 = bankDf.rdd.repartition(2);
	val bankLabelVectors = tempRdd1.map(transformToLabelVectors)
	bankLabelVectors.collect()
	
	println("\nTransformed Labeled Point :")
	val bankDf2 = spSession.createDataFrame(bankLabelVectors, classOf[LabeledPoint] )
	bankDf2.show(5)
	
	//Perform PCA
	import org.apache.spark.ml.feature.PCA
	val bankPCA = new PCA()
	bankPCA.setK(3)
	bankPCA.setInputCol("features")
	bankPCA.setOutputCol("pcaFeatures")
	val pcaModel = bankPCA.fit(bankDf2)
	val pcaResult = pcaModel.transform(bankDf2)
			.select("label","pcaFeatures")
			
	println("\nPCA Results:")
	pcaResult.show(5)
	
	//Split into training and testing data
	val Array(trainingData, testData) 
			= pcaResult.randomSplit(Array(0.7, 0.3))
	println("\nTraining Data count : " + trainingData.count())
	println("\nTest Data count : " + testData.count())
	
	import org.apache.spark.ml.classification.RandomForestClassifier
	import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

	//Create the model
	val rmClassifier = new RandomForestClassifier()
	rmClassifier.setLabelCol("label")
	rmClassifier.setFeaturesCol("pcaFeatures")
	val rmModel = rmClassifier.fit(trainingData)
	
	//Predict on the test data
	val predictions = rmModel.transform(testData)
	println("\nPredictions :")
	predictions.select("prediction","label","pcaFeatures").show(10)
	
	val evaluator = new MulticlassClassificationEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("label")
	evaluator.setMetricName("accuracy")
	println("\nAccuracy = " + evaluator.evaluate(predictions)  )
	
	println("\nConfusion Matrix:")
	predictions.groupBy("label","prediction").count().show()
}