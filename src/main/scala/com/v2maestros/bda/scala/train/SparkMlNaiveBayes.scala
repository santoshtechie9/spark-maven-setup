package com.v2maestros.bda.scala.train
/*
-----------------------------------------------------------------------------

           Naive Bayes : Spam Filtering
           
             Copyright : V2 Maestros @2016
                    
Problem Statement
*****************
The input data is a set of SMS messages that has been classified 
as either "ham" or "spam". The goal of the exercise is to build a
 model to identify messages as either ham or spam.

//// Techniques Used

1. Naive Bayes Classifier
2. Training and Testing
3. Confusion Matrix
4. Text Pre-Processing
5. Pipelines

-----------------------------------------------------------------------------
*/
object SparkMlNaiveBayes extends App{
	
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
	val smsData = spContext.textFile(datadir 
			+ "/SMSSpamCollection.csv")
	smsData.cache()
	smsData.count()
	
	//Remove the first line (contains headers)
	val firstLine=smsData.first()
	val dataLines = smsData.filter(x => x != firstLine)
	dataLines.count()
	
	import org.apache.spark.ml.linalg.{Vector, Vectors}
	import org.apache.spark.ml.feature.LabeledPoint
	import org.apache.spark.sql.Row;
	import org.apache.spark.sql.types._
	
	//Schema for Data Frame
	val schema =
	  StructType(
	    StructField("LABEL", DoubleType, false) ::
	    StructField("MESSAGE", StringType, false) :: Nil)
	
	def transformToNumeric( inputStr : String) : Row = {
	    val attList=inputStr.split(",")
	    
	   val smsType:Double = attList(0).contains("spam") match {
	            case  true => 1.0
	            case  false    => 0.0
	     }
	    
	    //Filter out columns not wanted at this stage
	    val values= Row(smsType, attList(1)
	                     )
	    return values
	 }   
	
	//Change to a Vector
	val smsVectors = dataLines.map(transformToNumeric)
	smsVectors.collect()

	println("Transformed data in Data Frame")
    val smsDf = spSession.createDataFrame(smsVectors, schema)
    smsDf.printSchema()
    smsDf.show(5)
    
   	//Split into training and testing data
	val Array(trainingData, testData) 
		= smsDf.randomSplit(Array(0.7, 0.3))
	trainingData.count()
	testData.count()
	
    //Setup pipeline
	import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
	import org.apache.spark.ml.Pipeline
	import org.apache.spark.ml.feature.{HashingTF, Tokenizer, IDF}
	
	//Setup tokenizer that splits sentences to words
	val tokenizer = new Tokenizer()
	tokenizer.setInputCol("MESSAGE")
	tokenizer.setOutputCol("WORDS")
	
	//Setup the TF compute function
	val hashingTF = new HashingTF()
	hashingTF.setInputCol(tokenizer.getOutputCol)
	hashingTF.setOutputCol("TEMPFEATURES")
	
	//Setup the IDF compute function
	val idf=new IDF()
	idf.setInputCol(hashingTF.getOutputCol)
	idf.setOutputCol("FEATURES")
	
	//Setup the Naive Bayes classifier
	val nbClassifier=new NaiveBayes()
	nbClassifier.setLabelCol("LABEL")
	nbClassifier.setFeaturesCol("FEATURES")
	nbClassifier.setPredictionCol("PREDICTION")
	
	//Setup the pipeline with all the transformers
	val pipeline = new Pipeline()
	pipeline.setStages(Array(tokenizer, 
					hashingTF,idf, nbClassifier))
	
	//Build the model
	val nbModel=pipeline.fit(trainingData)
	
	//Predict on the test data
	val prediction=nbModel.transform(testData)
	println("\nRaw Predictions : ")
	prediction.show()
	
	//Evaluate the precision of prediction
	import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
	val evaluator = new MulticlassClassificationEvaluator()
	evaluator.setPredictionCol("PREDICTION")
	evaluator.setLabelCol("LABEL")
	evaluator.setMetricName("accuracy")
	println("\nAccuracy = " + evaluator.evaluate(prediction) )
	
	//Print confusion matrix.
	println("\nConfusion Matrix : ")
	prediction.groupBy("LABEL","PREDICTION").count().show()
	
}