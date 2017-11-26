package com.v2maestros.bda.scala.train

/****************************************************************************

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Decision Trees

Problem Statement
*****************
The input data is the iris dataset. It contains recordings of 
information about flower samples. For each sample, the petal and 
sepal length and width are recorded along with the type of the 
flower. We need to use this dataset to build a decision tree 
model that can predict the type of flower based on the petal 
and sepal information.

## Techniques Used

1. Decision Trees 
2. Training and Testing
3. Confusion Matrix

*****************************************************************************/

object SparkMlDecisionTrees extends App {
	
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
  	println("\nLoading data file :")
	val irisData = spContext.textFile(datadir + "iris.csv")
	irisData.cache()
	irisData.take(5)
	
	//Remove the first line (contains headers)
	val dataLines = irisData.filter(x =>  !x.contains("Sepal"))
	dataLines.count()
	
	//Convert the RDD into a Dense Vector. As a part of this exercise
	//   1. Change labels to numeric ones
	
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

	println("\nTransformed data in Data Frame")
    val irisDf = spSession.createDataFrame(irisVectors, schema)
    irisDf.printSchema()
    irisDf.show(5)
   
	//Indexing needed as pre-req for Decision Trees
	import org.apache.spark.ml.feature.StringIndexer
	val stringIndexer = new StringIndexer()
	stringIndexer.setInputCol("SPECIES")
	stringIndexer.setOutputCol("INDEXED")
	val si_model = stringIndexer.fit(irisDf)
	val indexedIris = si_model.transform(irisDf)
	indexedIris.show()
	indexedIris.groupBy("SPECIES","INDEXED").count().show()
    
	println("\nCorrelation Analysis :")
   	for ( field <- schema.fields ) {
		if ( ! field.dataType.equals(StringType)) {
			println("Correlation between INDEXED and " + field.name +
					 " = " + indexedIris.stat.corr("INDEXED", field.name))
		}
	}
	
	//Transform to a Data Frame for input to Machine Learing
	//Drop columns that are not required (low correlation / strings)
	def transformToLabelVectors(inStr : Row ) : LabeledPoint = { 
	    val labelVectors = new LabeledPoint(
	    						inStr.getDouble(5) , 
								Vectors.dense(inStr.getDouble(1),
										inStr.getDouble(2),
										inStr.getDouble(3),
										inStr.getDouble(4)));
	    return labelVectors
	}
	val tempRdd1 = indexedIris.rdd.repartition(2);
	val irisLabelVectors 
		= tempRdd1.map(transformToLabelVectors)
	irisLabelVectors.collect()
	
	val irisDf2 = spSession.createDataFrame(
			irisLabelVectors, classOf[LabeledPoint] )
	
	println("\nData ready for ML")
	irisDf2.select("label","features").show(10)

	
	//Split into training and testing data
	val Array(trainingData, testData) 
		= irisDf2.randomSplit(Array(0.7, 0.3))
	trainingData.count()
	testData.count()
	
	import org.apache.spark.ml.classification.DecisionTreeClassifier
	import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
	import org.apache.spark.ml.feature.IndexToString

	//Create the model
	val dtClassifier = new DecisionTreeClassifier()
	dtClassifier.setMaxDepth(2)
	val dtModel = dtClassifier.fit(trainingData)
	
	dtModel.numNodes
	dtModel.depth
	
	val rawPredictions = dtModel.transform(testData)
	
	// Convert indexed labels back to original labels.
	val labelConverter = new IndexToString()
				  .setInputCol("label")
				  .setOutputCol("labelStr")
				  .setLabels(si_model.labels)
		
	val predConverter = new IndexToString()
				  .setInputCol("prediction")
				  .setOutputCol("predictionStr")
				  .setLabels(si_model.labels);

	
	val predictions = predConverter.transform(
			labelConverter.transform(rawPredictions))
	println("\nRaw predictions :")
	predictions.show()
	
	val evaluator = new MulticlassClassificationEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("label")
	evaluator.setMetricName("accuracy")
	println("\nAccuracy = " + evaluator.evaluate(predictions)  )
	
	println("\nConfusion Matrix:")
	//Draw confusion matrix
	predictions.groupBy("labelStr","predictionStr").count().show()
	
}