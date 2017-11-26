package com.v2maestros.bda.scala.train
/*
  				 Spark with Python

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Machine Learning - Clustering

The input data contains samples of cars and technical / price 
information about them. The goal of this problem is to group 
these cars into 4 clusters based on their attributes

//// Techniques Used

1. K-Means Clustering
2. Centering and Scaling

-----------------------------------------------------------------------------
*/
object SparkMlKMeansClustering extends App {
	
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
	val autoData = spContext.textFile(datadir 
						+ "/auto-data.csv")
	autoData.cache()
	autoData.count()
	
	//Remove the first line (contains headers)
	val firstLine=autoData.first()
	val dataLines = autoData.filter(x => x != firstLine)
	dataLines.count()
	
	import org.apache.spark.ml.linalg.{Vector, Vectors}
	import org.apache.spark.ml.feature.LabeledPoint
	import org.apache.spark.sql.Row;
	import org.apache.spark.sql.types._
	
	//Schema for Data Frame
	val schema =
	  StructType(
	    StructField("DOORS", DoubleType, false) ::
	    StructField("BODY", DoubleType, false) ::
	    StructField("HP", DoubleType, false) ::
	    StructField("RPM", DoubleType, false) ::
	    StructField("MPG", DoubleType, false) :: Nil)
	    
	//Convert to Local Vector.
	def transformToNumeric( inputStr : String) : Row = {
	    val attList=inputStr.split(",")
	
	    val doors = attList(3).contains("two") match {
	        case  true => 0.0
	        case  false    => 1.0
	    }
	    val body = attList(4).contains("sedan") match {
	        case  true => 0.0
	        case  false    => 1.0
	    }     
	    //Filter out columns not wanted at this stage
	    //only use doors, body, hp, rpm, mpg-city
	    val values= Row( doors, body,
	                     attList(7).toDouble, attList(8).toDouble,
	                     attList(9).toDouble)
	    return values
	}
	//Change to a Vector
	val autoVectors = dataLines.map(transformToNumeric)
	autoVectors.collect()

	println("Transformed data in Data Frame")
    val autoDf = spSession.createDataFrame(autoVectors, schema)
    autoDf.printSchema()
    autoDf.show(5)
    
    /*--------------------------------------------------------------------------
	Prepare for Machine Learning - Perform Centering and Scaling
	--------------------------------------------------------------------------*/
    
    val meanVal = autoDf.agg(avg("DOORS"), avg("BODY"),avg("HP"),
    		avg("RPM"),avg("MPG")).collectAsList().get(0)
    		
    val stdVal = autoDf.agg(stddev("DOORS"), stddev("BODY"),
    		stddev("HP"),stddev("RPM"),stddev("MPG")).collectAsList().get(0)
    		
    val bcMeans=spContext.broadcast(meanVal)
	val bcStdDev=spContext.broadcast(stdVal)
	
	def centerAndScale(inRow : Row ) : LabeledPoint  = {
	    val meanArray=bcMeans.value
	    val stdArray=bcStdDev.value
	    
	    var retArray=Array[Double]()
	    
	    for (i <- 0 to inRow.size - 1)  {
	    	val csVal = ( inRow.getDouble(i) - meanArray.getDouble(i)) /
	    					 stdArray.getDouble(i)
	        retArray = retArray :+ csVal
	    }
	    return  new LabeledPoint(1.0,Vectors.dense(retArray))
	} 
	
    val tempRdd1 = autoDf.rdd.repartition(2);
	val autoCSRDD = tempRdd1.map(centerAndScale)
	autoCSRDD.collect()
	 
	val autoDf2 = spSession.createDataFrame(autoCSRDD, classOf[LabeledPoint] )
		
	println("Data ready for ML")
	autoDf2.select("label","features").show(10)
	
	import  org.apache.spark.ml.clustering.KMeans
	val kmeans = new KMeans()
	kmeans.setK(4)
	kmeans.setSeed(1L)
	
	//Perform K-Means Clustering
	val model = kmeans.fit(autoDf2)
	val predictions = model.transform(autoDf2)
	
	println("Groupings :")
	predictions.select("features","prediction").show()
	predictions.groupBy("prediction").count().show()
	
}